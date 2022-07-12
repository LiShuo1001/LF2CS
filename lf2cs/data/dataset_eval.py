import os
import random
from .my_dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from .class_balanced_sampler import ClassBalancedSampler, ClassBalancedSamplerTest


class EvalDataset(Dataset):

    def __init__(self, task, image_features, split='train'):
        self.task = task
        self.split = split
        self.image_features = image_features

        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        pass

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image = self.image_features[self.image_roots[idx]]
        label = self.labels[idx]
        return image, label

    @staticmethod
    def folders(data_root):
        train_folder = os.path.join(data_root, MyDataset.dataset_split_train)
        val_folder = os.path.join(data_root, MyDataset.dataset_split_val)
        test_folder = os.path.join(data_root, MyDataset.dataset_split_test)

        folders_train = [os.path.join(train_folder, label) for label in os.listdir(train_folder)
                         if os.path.isdir(os.path.join(train_folder, label))]
        folders_val = [os.path.join(val_folder, label) for label in os.listdir(val_folder)
                       if os.path.isdir(os.path.join(val_folder, label))]
        folders_test = [os.path.join(test_folder, label) for label in os.listdir(test_folder)
                        if os.path.isdir(os.path.join(test_folder, label))]

        random.seed(1)
        random.shuffle(folders_train)
        random.shuffle(folders_val)
        random.shuffle(folders_test)
        return folders_train, folders_val, folders_test

    @staticmethod
    def get_data_loader(task, image_features, num_per_class=1, split="train", sampler_test=False, shuffle=False):
        if split == "train":
            sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
        else:
            if not sampler_test:
                sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
            else:  # test
                sampler = ClassBalancedSamplerTest(task.num_classes, task.test_num, shuffle=shuffle)
                pass
            pass

        dataset = EvalDataset(task, image_features=image_features, split=split)
        return DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    pass

