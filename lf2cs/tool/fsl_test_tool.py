import torch
import numpy as np
from ..data.task import Task
from .util_tools import Tools
from .runner_tool import RunnerTool
from ..data.dataset_test import TestDataset


class FSLTestTool(object):

    def __init__(self, model_fn, data_root, num_way=5, num_shot=1,
                 episode_size=15, test_episode=600, transform=None, txt_path=None):
        self.model_fn = model_fn
        self.transform = transform
        self.txt_path = txt_path

        self.folders_train, self.folders_val, self.folders_test = TestDataset.folders(data_root)

        self.test_episode = test_episode
        self.num_way = num_way
        self.num_shot = num_shot
        self.episode_size = episode_size
        pass

    @staticmethod
    def _compute_confidence_interval(data):
        a = 1.0 * np.array(data)
        m = np.mean(a)
        std = np.std(a)
        pm = 1.96 * (std / np.sqrt(len(a)))
        return m, pm

    def eval(self, num_way=5, num_shot=1, episode_size=15, test_episode=1000):
        acc_list = self._val_no_mean(self.folders_test, sampler_test=True, num_way=num_way,
                                     num_shot=num_shot, episode_size=episode_size, test_episode=test_episode)
        m, pm = self._compute_confidence_interval(acc_list)
        return m, pm

    def val_train(self):
        return self._val(self.folders_train, sampler_test=False, all_episode=self.test_episode)

    def val_val(self):
        return self._val(self.folders_val, sampler_test=False, all_episode=self.test_episode)

    def val_test(self):
        return self._val(self.folders_test, sampler_test=False, all_episode=self.test_episode)

    def val_test2(self):
        return self._val(self.folders_test, sampler_test=True, all_episode=self.test_episode)

    def test(self, test_avg_num, episode=0, is_print=True):
        acc_list = []
        for _ in range(test_avg_num):
            acc = self._val(self.folders_test, sampler_test=True, all_episode=self.test_episode)
            acc_list.append(acc)
            pass

        mean_acc = np.mean(acc_list)
        if is_print:
            for acc in acc_list:
                Tools.print("epoch={}, Test accuracy={}".format(episode, acc), txt_path=self.txt_path)
                pass
            Tools.print("epoch={}, Mean Test accuracy={}".format(episode, mean_acc), txt_path=self.txt_path)
            pass
        return mean_acc

    def val(self, episode=0, is_print=True, has_test=True):
        acc_train = self.val_train()
        if is_print:
            Tools.print("Train {} Accuracy: {}".format(episode, acc_train), txt_path=self.txt_path)

        acc_val = self.val_val()
        if is_print:
            Tools.print("Val   {} Accuracy: {}".format(episode, acc_val), txt_path=self.txt_path)

        acc_test1 = self.val_test()
        if is_print:
            Tools.print("Test1 {} Accuracy: {}".format(episode, acc_test1), txt_path=self.txt_path)

        # if has_test:
        #     acc_test2 = self.val_test2()
        #     if is_print:
        #         Tools.print("Test2 {} Accuracy: {}".format(episode, acc_test2), txt_path=self.txt_path)
        #         pass
        return acc_val

    def _val(self, folders, sampler_test, all_episode):
        accuracies = self._val_no_mean(folders, sampler_test, num_way=self.num_way, num_shot=self.num_shot,
                                       episode_size=self.episode_size, test_episode=all_episode)
        return np.mean(np.array(accuracies, dtype=np.float))

    def _val_no_mean(self, folders, sampler_test, num_way, num_shot, episode_size, test_episode):
        accuracies = []
        for i in range(test_episode):
            total_rewards = 0
            counter = 0

            task = Task(folders, num_way, num_shot, episode_size)
            sample_data_loader = TestDataset.get_data_loader(task, num_shot, "train", sampler_test=sampler_test,
                                                           shuffle=False, transform=self.transform)
            batch_data_loader = TestDataset.get_data_loader(task, 15, "val", sampler_test=sampler_test,
                                                          shuffle=True, transform=self.transform)
            samples, labels = sample_data_loader.__iter__().next()

            with torch.no_grad():
                samples = RunnerTool.to_cuda(samples)
                for batches, batch_labels in batch_data_loader:
                    results = self.model_fn(samples, RunnerTool.to_cuda(batches), num_way=num_way, num_shot=num_shot)

                    _, predict_labels = torch.max(results.data, 1)
                    batch_size = batch_labels.shape[0]
                    rewards = [1 if predict_labels[j].cpu() == batch_labels[j] else 0 for j in range(batch_size)]
                    total_rewards += np.sum(rewards)

                    counter += batch_size
                    pass
                pass

            accuracies.append(total_rewards / 1.0 / counter)
            pass
        return accuracies

    pass
