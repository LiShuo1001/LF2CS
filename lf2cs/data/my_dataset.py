import os
import platform
from PIL import Image


class MyDataset(object):

    dataset_name_miniimagenet = "miniimagenet"
    dataset_name_tieredimagenet = "tieredimagenet"
    dataset_name_cifarfs = "CIFARFS"
    dataset_name_fc100 = "FC100"
    dataset_name_omniglot = "omniglot"

    dataset_split_train = "train"
    dataset_split_val = "val"
    dataset_split_test = "test"

    @staticmethod
    def get_data_split(data_root, split="train"):
        train_folder = os.path.join(data_root, split)

        if not os.path.exists(train_folder) or len(os.listdir(train_folder)) <= 0:
            raise Exception("Refer to the structure of Omniglot, please prepare data in {}.".format(data_root))

        count_image, count_class, data_train_list = 0, 0, []
        for label in os.listdir(train_folder):
            now_class_path = os.path.join(train_folder, label)
            if os.path.isdir(now_class_path):
                for name in os.listdir(now_class_path):
                    data_train_list.append((count_image, count_class, os.path.join(now_class_path, name)))
                    count_image += 1
                    pass
                count_class += 1
            pass

        if count_image < 5 or count_class < 5:
            raise Exception("Refer to the structure of Omniglot, please prepare data in {}.".format(data_root))

        return data_train_list

    @staticmethod
    def read_image(image_path, transform=None):
        image = Image.open(image_path).convert('RGB')
        if transform is not None:
            image = transform(image)
        return image

    @staticmethod
    def get_data_root(dataset_name, is_png=True):
        if "Linux" in platform.platform():
            data_root = "./data/{}".format(dataset_name)
            if not os.path.isdir(data_root):
                raise Exception("{} not exist".format(dataset_name))
        else:
            raise Exception("platform error")

        if dataset_name == MyDataset.dataset_name_miniimagenet:
            data_root = os.path.join(data_root, "png") if is_png else data_root
        return data_root

    @staticmethod
    def get_ways_shots(dataset_name, split):
        if dataset_name == MyDataset.dataset_name_miniimagenet \
                or dataset_name == MyDataset.dataset_name_cifarfs \
                or dataset_name == MyDataset.dataset_name_fc100:
            if split == MyDataset.dataset_split_test:
                ways = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                shots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            elif split == MyDataset.dataset_split_val:
                ways = [2, 5, 10, 15, 16]
                shots = [1, 5, 10, 15, 20, 30, 40, 50]
            else:
                ways = [2, 5, 10, 15, 20, 30, 40, 50]
                shots = [1, 5, 10, 15, 20, 30, 40, 50]
        elif dataset_name == MyDataset.dataset_name_tieredimagenet:
            if split == MyDataset.dataset_split_test:
                ways = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
                shots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            elif split == MyDataset.dataset_split_val:
                ways = [2, 5, 10, 15, 20, 30, 40, 50]
                shots = [1, 5, 10, 15, 20, 30, 40, 50]
            else:
                ways = [2, 5, 10, 15, 20, 30, 40, 50]
                shots = [1, 5, 10, 15, 20, 30, 40, 50]
        elif dataset_name == MyDataset.dataset_name_omniglot:
            if split == MyDataset.dataset_split_test:
                ways = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
                shots = [1, 2, 3, 4, 5]
            elif split == MyDataset.dataset_split_val:
                ways = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
                shots = [1, 2, 3, 4, 5]
            else:
                ways = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
                shots = [1, 2, 3, 4, 5]
        else:
            raise Exception(".")
        return ways, shots

    pass
