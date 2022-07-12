import torch
import numpy as np
from ..data.task import Task
from .runner_tool import RunnerTool
from ..data.my_dataset import MyDataset
from ..data.dataset_eval import EvalDataset
from ..data.dataset_test import TestDataset


class FSLEvalTool(object):

    def __init__(self, model_fn, data_root, image_features, num_way=5, num_shot=1,
                 episode_size=15, test_episode=600, txt_path=None):
        self.model_fn = model_fn
        self.image_features = image_features
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

    def eval(self, num_way=5, num_shot=1, episode_size=15, test_episode=1000, split=MyDataset.dataset_split_test):
        folders_test = self.folders_test
        if split == MyDataset.dataset_split_train:
            folders_test = self.folders_train
        elif split == MyDataset.dataset_split_val:
            folders_test = self.folders_val
            pass

        acc_list = self._val_no_mean(folders_test, sampler_test=True, num_way=num_way,
                                     num_shot=num_shot, episode_size=episode_size, test_episode=test_episode)
        m, pm = self._compute_confidence_interval(acc_list)
        return m, pm

    def _val_no_mean(self, folders, sampler_test, num_way, num_shot, episode_size, test_episode):
        accuracies = []
        for i in range(test_episode):
            total_rewards = 0
            counter = 0

            task = Task(folders, num_way, num_shot, episode_size)
            sample_data_loader = EvalDataset.get_data_loader(task, self.image_features, num_shot, "train",
                                                             sampler_test=sampler_test, shuffle=False)
            num_per_class = 5 if num_shot > 1 else 3
            batch_data_loader = EvalDataset.get_data_loader(task, self.image_features, num_per_class, "val",
                                                            sampler_test=sampler_test, shuffle=True)
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

