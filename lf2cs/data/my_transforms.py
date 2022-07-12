import numpy as np
from .my_dataset import MyDataset
import torchvision.transforms as transforms


class MyTransforms(object):

    @staticmethod
    def get_transform_miniimagenet(normalize, has_lf2cs=True, is_fsl_simple=True, is_css=False):
        transform_train_lf2cs = transforms.Compose([
            # transforms.RandomCrop(84, padding=8),
            transforms.RandomResizedCrop(size=84, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

        transform_train_fsl_simple = transforms.Compose([
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        transform_train_fsl_hard = transforms.Compose([
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        transform_train_fsl_css = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(84),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        transform_train_fsl = transform_train_fsl_simple if is_fsl_simple else transform_train_fsl_hard
        transform_train_fsl = transform_train_fsl_css if is_css else transform_train_fsl

        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        if has_lf2cs:
            return transform_train_lf2cs, transform_train_fsl, transform_test
        else:
            return transform_train_fsl, transform_test
        pass

    @classmethod
    def get_transform_tieredimagenet(cls, normalize, has_lf2cs=True, is_fsl_simple=True, is_css=False):
        return cls.get_transform_miniimagenet(normalize, has_lf2cs, is_fsl_simple, is_css)

    @staticmethod
    def get_transform_cifar(normalize, has_lf2cs=True, is_fsl_simple=True, is_css=False, size=32):
        change = transforms.Resize(size) if size > 32 else lambda x: x
        transform_train_lf2cs = transforms.Compose([
            change, transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize
        ])

        transform_train_fsl_simple = transforms.Compose([
            change, transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        transform_train_fsl_hard = transforms.Compose([
            change, transforms.RandomResizedCrop(size=size),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        transform_train_fsl_css = transforms.Compose([
            change, transforms.RandomRotation(20),
            transforms.RandomResizedCrop(size=size),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        transform_train_fsl = transform_train_fsl_simple if is_fsl_simple else transform_train_fsl_hard
        transform_train_fsl = transform_train_fsl_css if is_css else transform_train_fsl

        transform_test = transforms.Compose([change, transforms.ToTensor(), normalize])

        if has_lf2cs:
            return transform_train_lf2cs, transform_train_fsl, transform_test
        else:
            return transform_train_fsl, transform_test
        pass

    @staticmethod
    def get_transform_omniglot(normalize, has_lf2cs=True, size=28):
        transform_train_lf2cs = transforms.Compose([transforms.RandomRotation(30, fill=255),
                                                 transforms.Resize(size),
                                                 transforms.RandomCrop(size, padding=4, fill=255),
                                                 transforms.ToTensor(), normalize])
        transform_train_fsl = transforms.Compose([transforms.RandomRotation(30, fill=255),
                                                  transforms.Resize(size),
                                                  transforms.RandomCrop(size, padding=4, fill=255),
                                                  transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), normalize])

        if has_lf2cs:
            return transform_train_lf2cs, transform_train_fsl, transform_test
        else:
            return transform_train_fsl, transform_test
        pass

    @classmethod
    def get_transform(cls, dataset_name, has_lf2cs=True, is_fsl_simple=True, is_css=False, cifar_size=32, omniglot_size=28):
        normalize_1 = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                             std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        normalize_2 = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                              np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        normalize_3 = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        normalize_4 = transforms.Normalize(mean=[0.92206], std=[0.08426])
        if dataset_name == MyDataset.dataset_name_miniimagenet:
            return cls.get_transform_miniimagenet(normalize_1, has_lf2cs=has_lf2cs,
                                                  is_fsl_simple=is_fsl_simple, is_css=is_css)
        elif dataset_name == MyDataset.dataset_name_tieredimagenet:
            return cls.get_transform_tieredimagenet(normalize_1, has_lf2cs=has_lf2cs,
                                                    is_fsl_simple=is_fsl_simple, is_css=is_css)
        elif dataset_name == MyDataset.dataset_name_cifarfs or dataset_name == MyDataset.dataset_name_fc100:
            return cls.get_transform_cifar(normalize_3, has_lf2cs=has_lf2cs,
                                           is_fsl_simple=is_fsl_simple, is_css=is_css, size=cifar_size)
        elif dataset_name == MyDataset.dataset_name_omniglot:
            return cls.get_transform_omniglot(normalize_4, has_lf2cs=has_lf2cs, size=omniglot_size)
        else:
            raise Exception("......")
        pass

    pass

