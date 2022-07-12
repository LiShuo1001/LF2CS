import os
import sys
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.append("./lf2cs")
from lf2cs.core.produce_class import ProduceClass
from lf2cs.data.my_dataset import MyDataset
from lf2cs.data.dataset_train import TrainDataset
from lf2cs.net.Normalize import Normalize
from lf2cs.net.LF2CSNet import LF2CSNet
from lf2cs.tool.util_tools import Tools
from lf2cs.tool.fsl_test_tool import FSLTestTool
from lf2cs.tool.lf2cs_test_tool import LF2CSTestTool
from lf2cs.tool.runner_tool import RunnerTool


class Runner(object):

    def __init__(self, config):
        self.config = config

        # all data
        self.data_train = MyDataset.get_data_split(self.config.data_root, split=MyDataset.dataset_split_train)
        self.task_train = TrainDataset(
            self.data_train, self.config.num_way, self.config.num_shot, transform_train_lf2cs=self.config.transform_train_lf2cs,
            transform_train_fsl=self.config.transform_train_fsl, transform_test=self.config.transform_test)
        self.task_train_loader = DataLoader(self.task_train, self.config.batch_size, True, num_workers=self.config.num_workers)

        # lf2cs
        self.produce_class = ProduceClass(len(self.data_train), self.config.lf2cs_out_dim)
        self.produce_class.init()
        self.task_train.set_samples_class(self.produce_class.classes)

        # model
        self.norm = Normalize(2)
        self.model = RunnerTool.to_cuda(LF2CSNet(low_dim=self.config.lf2cs_out_dim,
                                                 fsl_dim=self.config.fsl_out_dim, encoder=self.config.encoders))
        RunnerTool.to_cuda(self.model.apply(RunnerTool.weights_init))

        # optim
        self.optim = torch.optim.SGD(self.model.parameters(),
                                     lr=self.config.learning_rate, momentum=0.9, weight_decay=5e-4)

        # loss
        self.lf2cs_loss = RunnerTool.to_cuda(nn.CrossEntropyLoss())
        self.fsl_loss = RunnerTool.to_cuda(nn.MSELoss())

        # Eval
        self.test_tool_fsl = FSLTestTool(self.matching_test, data_root=self.config.data_root,
                                         num_way=self.config.num_way, num_shot=self.config.num_shot,
                                         episode_size=self.config.episode_size, test_episode=self.config.test_episode,
                                         transform=self.task_train.transform_test, txt_path=self.config.log_file)
        self.test_tool_lf2cs = LF2CSTestTool(feature_encoder=None, lf2cs_model=self.model,
                                             data_root=self.config.data_root, batch_size=self.config.batch_size,
                                             num_workers=self.config.num_workers, lf2cs_out_dim=self.config.lf2cs_out_dim,
                                             transform=self.task_train.transform_test,
                                             txt_path=self.config.log_file, k=self.config.knn_k)
        pass

    def load_model(self, checkpoint=None):
        net_dir = checkpoint if checkpoint else self.config.net_dir

        if os.path.exists(net_dir):
            checkpoint = torch.load(net_dir)
            if "module." in list(checkpoint.keys())[0]:
                checkpoint = {key.replace("module.", ""): checkpoint[key] for key in checkpoint}
            try:
                self.model.load_state_dict(checkpoint)
            except Exception:
                result = self.model.load_state_dict(checkpoint, strict=False)
                Tools.print(result)
                pass
            Tools.print("load model success from {}".format(net_dir), txt_path=self.config.log_file)
            pass
        pass

    def matching(self, task_data):
        data_batch_size, data_image_num, data_num_channel, data_width, data_weight = task_data.shape
        data_x = task_data.view(-1, data_num_channel, data_width, data_weight)
        net_out, lf2cs_out, lf2cs_norm = self.model(data_x)
        z = net_out.view(data_batch_size, data_image_num, -1)

        # features
        z_support, z_query = z.split(self.config.num_shot * self.config.num_way, dim=1)
        z_batch_size, z_num, z_dim = z_support.shape
        z_support = z_support.view(z_batch_size, self.config.num_way * self.config.num_shot, z_dim)
        z_query_expand = z_query.expand(z_batch_size, self.config.num_way * self.config.num_shot, z_dim)

        # similarity
        z_support = self.norm(z_support)
        similarities = torch.sum(z_support * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(z_batch_size, self.config.num_way, self.config.num_shot)
        predicts = torch.mean(similarities, dim=-1)

        # LF2CS
        lf2cs_out = lf2cs_out.view(data_batch_size, data_image_num, -1)[:, -1, :]
        lf2cs_norm = lf2cs_norm.view(data_batch_size, data_image_num, -1)[:, -1, :]
        return predicts, lf2cs_out, lf2cs_norm

    def matching_test(self, samples, batches, num_way, num_shot):
        batch_num, _, _, _ = batches.shape

        sample_z, _, _ = self.model(samples)  # 5x64*5*5
        batch_z, _, _ = self.model(batches)  # 75x64*5*5
        z_support = sample_z.view(num_way * num_shot, -1)
        z_query = batch_z.view(batch_num, -1)
        _, z_dim = z_query.shape

        z_support_expand = z_support.unsqueeze(0).expand(batch_num, num_way * num_shot, z_dim)
        z_query_expand = z_query.unsqueeze(1).expand(batch_num, num_way * num_shot, z_dim)

        # similarity
        z_support_expand = self.norm(z_support_expand)
        similarities = torch.sum(z_support_expand * z_query_expand, -1)
        similarities = torch.softmax(similarities, dim=1)
        similarities = similarities.view(batch_num, num_way, num_shot)
        predicts = torch.mean(similarities, dim=-1)
        return predicts

    def train(self):
        Tools.print()
        best_accuracy = 0.0
        Tools.print("Training...", txt_path=self.config.log_file)

        # Init Update
        try:
            self.model.eval()
            Tools.print("Init label {} .......", txt_path=self.config.log_file)
            self.produce_class.reset()
            for task_data, task_labels, task_index, task_ok in tqdm(self.task_train_loader):
                lf2cs_labels = RunnerTool.to_cuda(task_index[:, -1])
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)
                _, _, lf2cs_out_l2norm = self.model(task_data[:, -1])
                self.produce_class.cal_label(lf2cs_out_l2norm, lf2cs_labels)
                pass
            Tools.print("Epoch: {}/{}".format(self.produce_class.count, self.produce_class.count_2),
                        txt_path=self.config.log_file)
        finally:
            pass

        for epoch in range(0, self.config.train_epoch):
            self.model.train()

            Tools.print()
            lr= self.config.adjust_learning_rate(self.optim, epoch, self.config.first_epoch,
                                                     self.config.t_epoch, self.config.learning_rate)
            Tools.print('Epoch: [{}] lr={}'.format(epoch, lr), txt_path=self.config.log_file)

            self.produce_class.reset()
            Tools.print(self.task_train.classes)
            is_ok_total, is_ok_acc = 0, 0
            all_loss, all_loss_fsl, all_loss_lf2cs = 0.0, 0.0, 0.0
            for task_data, task_labels, task_index, task_ok in tqdm(self.task_train_loader):
                lf2cs_labels = RunnerTool.to_cuda(task_index[:, -1])
                task_data, task_labels = RunnerTool.to_cuda(task_data), RunnerTool.to_cuda(task_labels)

                ###########################################################################
                # 1 calculate features
                relations, lf2cs_out_logits, lf2cs_out_l2norm = self.matching(task_data)

                # 2
                lf2cs_targets = self.produce_class.get_label(lf2cs_labels)
                self.produce_class.cal_label(lf2cs_out_l2norm, lf2cs_labels)

                # 3 loss
                loss_fsl = self.fsl_loss(relations, task_labels)
                loss_lf2cs = self.lf2cs_loss(lf2cs_out_logits, lf2cs_targets)
                loss = loss_fsl + loss_lf2cs
                all_loss += loss.item()
                all_loss_fsl += loss_fsl.item()
                all_loss_lf2cs += loss_lf2cs.item()

                # 4 backward
                self.model.zero_grad()
                loss.backward()
                self.optim.step()

                # is ok
                is_ok_acc += torch.sum(torch.cat(task_ok))
                is_ok_total += torch.prod(torch.tensor(torch.cat(task_ok).shape))
                ###########################################################################
                pass

            ###########################################################################
            # print
            Tools.print("{:6} loss:{:.3f} fsl:{:.3f} lf2cs:{:.3f} ok:{:.3f}({}/{})".format(
                epoch, all_loss / len(self.task_train_loader),
                all_loss_fsl / len(self.task_train_loader), all_loss_lf2cs / len(self.task_train_loader),
                int(is_ok_acc) / int(is_ok_total), is_ok_acc, is_ok_total), txt_path=self.config.log_file)
            Tools.print("Train: [{}] {}/{}".format(epoch, self.produce_class.count, self.produce_class.count_2), txt_path=self.config.log_file)
            ###########################################################################

            ###########################################################################
            # Val
            if (epoch + 1) % self.config.val_freq == 0:
                self.model.eval()

                self.test_tool_lf2cs.val(epoch=epoch)
                val_accuracy = self.test_tool_fsl.val(episode=epoch, is_print=True, has_test=False)

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(self.model.state_dict(), Tools.new_dir(self.config.net_dir))
                    Tools.print("Save networks for epoch: {}".format(epoch), txt_path=self.config.log_file)
                    pass
                pass
            ###########################################################################
            pass

        pass

    pass


def train(config):
    runner = Runner(config=config)

    # init eval
    # runner.model.eval()
    # runner.test_tool_lf2cs.val(epoch=0, is_print=True)
    # runner.test_tool_fsl.val(episode=0, is_print=True, has_test=False)

    # train
    runner.train()

    # load best model
    runner.load_model()
    runner.model.eval()
    # final eval: for training
    # runner.test_tool_lf2cs.val(epoch=config.train_epoch, is_print=True)
    # runner.test_tool_fsl.val(episode=config.train_epoch, is_print=True)
    # final eval: for paper
    runner.test_tool_fsl.test(test_avg_num=5, episode=config.train_epoch, is_print=True)
    pass


def test(config, checkpoint=None):
    runner = Runner(config=config)

    # load model
    runner.load_model(checkpoint=checkpoint)
    runner.model.eval()
    # eval
    # runner.test_tool_lf2cs.val(epoch=config.train_epoch, is_print=True)
    # runner.test_tool_fsl.val(episode=config.train_epoch, is_print=True)
    runner.test_tool_fsl.test(test_avg_num=10, episode=config.train_epoch, is_print=True)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LF2CS.')
    parser.add_argument('-gpuid', type=int, default=0, help='gpu id: default 0')
    parser.add_argument('-type', type=str, default="test", help='run type: train, test')
    parser.add_argument('-dataset', type=str, default=MyDataset.dataset_name_omniglot,
                        help='dataset name: {}, {}, {}'.format(
                            MyDataset.dataset_name_omniglot, MyDataset.dataset_name_miniimagenet,
                            MyDataset.dataset_name_tieredimagenet))
    parser.add_argument('-checkpoint', type=str, default="./models/{}/our/our.pkl".format(
        MyDataset.dataset_name_omniglot), help='checkpoint')
    args = parser.parse_args()

    if args.dataset == MyDataset.dataset_name_omniglot:
        from configs_omniglot import Config

        now_config = Config(gpu_id=args.gpuid)
    else:
        from configs_others import Config

        now_config = Config(gpu_id=args.gpuid, dataset_name=args.dataset, is_conv_4=True)
        pass

    if args.type == "train":
        train(now_config)
    else:
        test(now_config, checkpoint=args.checkpoint)
        pass

    pass

