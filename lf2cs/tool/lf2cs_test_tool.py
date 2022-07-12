import torch
import numpy as np
from .util_tools import Tools
from collections import Counter
from .runner_tool import RunnerTool
from torch.utils.data import DataLoader
from ..data.my_dataset import MyDataset
from ..data.dataset_lf2cs import LF2CSDataset


class KNN(object):

    @classmethod
    def cal(cls, labels, dist, train_labels, max_c, k, t):
        # ---------------------------------------------------------------------------------- #
        batch_size = labels.size(0)
        yd, yi = dist.topk(k + 1, dim=1, largest=True, sorted=True)
        yd, yi = yd[:, 1:], yi[:, 1:]
        retrieval = train_labels[yi]

        retrieval_1_hot = RunnerTool.to_cuda(torch.zeros(k, max_c)).resize_(batch_size * k, max_c).zero_().scatter_(
            1, retrieval.view(-1, 1), 1).view(batch_size, -1, max_c)
        yd_transform = yd.clone().div_(t).exp_().view(batch_size, -1, 1)
        probs = torch.sum(torch.mul(retrieval_1_hot, yd_transform), 1)
        _, predictions = probs.sort(1, True)
        # ---------------------------------------------------------------------------------- #

        correct = predictions.eq(labels.data.view(-1, 1))

        top1 = correct.narrow(1, 0, 1).sum().item()
        top5 = correct.narrow(1, 0, 5).sum().item()
        return top1, top5

    @classmethod
    def knn(cls, feature_encoder, lf2cs_model, low_dim, train_loader, k, t=0.1):

        with torch.no_grad():
            n_sample = train_loader.dataset.__len__()
            out_memory = RunnerTool.to_cuda(torch.zeros(n_sample, low_dim).t())
            train_labels = RunnerTool.to_cuda(torch.LongTensor(train_loader.dataset.train_label))
            max_c = train_labels.max() + 1

            # clustering 1
            clustering = np.zeros(n_sample, dtype=np.int)

            out_list = []
            for batch_idx, (inputs, labels, indexes) in enumerate(train_loader):
                inputs = RunnerTool.to_cuda(inputs)

                if feature_encoder is None:
                    _, _, out_l2norm = lf2cs_model(inputs)
                else:
                    features = feature_encoder(inputs)  # 5x64*19*19
                    _, _, out_l2norm = lf2cs_model(features)
                    pass

                # clustering 2
                now_clustering = torch.argmax(out_l2norm, dim=1).cpu()
                clustering[indexes] = now_clustering

                out_list.append([out_l2norm, RunnerTool.to_cuda(labels)])
                out_memory[:, batch_idx * inputs.size(0):(batch_idx + 1) * inputs.size(0)] = out_l2norm.data.t()
                pass

            top1, top5, total = 0., 0., 0
            for out in out_list:
                dist = torch.mm(out[0], out_memory)
                _top1, _top5 = cls.cal(out[1], dist, train_labels, max_c, k, t)

                top1 += _top1
                top5 += _top5
                total += out[1].size(0)
                pass

            # clustering 3
            acc_cluster = cls.cluster_acc(clustering, train_labels.cpu().long())

            return top1 / total, top5 / total, acc_cluster

        pass

    @staticmethod
    def cluster_acc(clustering, train_labels):
        counter_dict = {}
        for index, value in enumerate(clustering):
            if value not in counter_dict:
                counter_dict[value] = []
            counter_dict[value].append(int(train_labels[index]))
            pass
        for key in counter_dict:
            counter_dict[key] = dict(Counter(counter_dict[key]))
            pass
        return 0

    pass


class LF2CSTestTool(object):

    def __init__(self, feature_encoder, lf2cs_model, data_root, transform,
                 batch_size=64, num_workers=8, lf2cs_out_dim=512, k=100, txt_path=None):
        self.feature_encoder = feature_encoder if feature_encoder is None else RunnerTool.to_cuda(feature_encoder)
        self.lf2cs_model = RunnerTool.to_cuda(lf2cs_model)
        self.lf2cs_out_dim = lf2cs_out_dim
        self.k = k
        self.txt_path = txt_path

        self.data_train = MyDataset.get_data_split(data_root, split=MyDataset.dataset_split_train)
        self.train_loader = DataLoader(LF2CSDataset(self.data_train, transform), batch_size, False, num_workers=num_workers)

        self.data_val = MyDataset.get_data_split(data_root, split=MyDataset.dataset_split_val)
        self.val_loader = DataLoader(LF2CSDataset(self.data_val, transform), batch_size, False, num_workers=num_workers)

        self.data_test = MyDataset.get_data_split(data_root, split=MyDataset.dataset_split_test)
        self.test_loader = DataLoader(LF2CSDataset(self.data_test, transform), batch_size, False, num_workers=num_workers)
        pass

    def val_lf2cs(self, lf2cs_loader):
        acc_1, acc_2, acc_3 = KNN.knn(self.feature_encoder, self.lf2cs_model, self.lf2cs_out_dim, lf2cs_loader, self.k)
        return acc_1, acc_2, acc_3

    def val(self, epoch, is_print=True):
        if is_print:
            Tools.print()
            Tools.print("Test {} .......".format(epoch), txt_path=self.txt_path)
            pass

        acc_1_train, acc_2_train, acc_3_train = self.val_lf2cs(lf2cs_loader=self.train_loader)
        if is_print:
            Tools.print("Epoch: {} Train {:.4f}/{:.4f} {:.4f}".format(epoch, acc_1_train, acc_2_train, acc_3_train), txt_path=self.txt_path)

        acc_1_val, acc_2_val, acc_3_val = self.val_lf2cs(lf2cs_loader=self.val_loader)
        if is_print:
            Tools.print("Epoch: {} Val   {:.4f}/{:.4f} {:.4f}".format(epoch, acc_1_val, acc_2_val, acc_3_val), txt_path=self.txt_path)

        acc_1_test, acc_2_test, acc_3_test = self.val_lf2cs(lf2cs_loader=self.test_loader)
        if is_print:
            Tools.print("Epoch: {} Test  {:.4f}/{:.4f} {:.4f}".format(epoch, acc_1_test, acc_2_test, acc_3_test), txt_path=self.txt_path)
            pass
        return acc_1_val

    pass

