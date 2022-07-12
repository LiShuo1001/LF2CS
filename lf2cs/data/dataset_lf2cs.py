from PIL import Image
from torch.utils.data import Dataset


class LF2CSDataset(Dataset):

    def __init__(self, data_list, transform):
        self.data_list = data_list
        self.train_label = [one[1] for one in self.data_list]
        self.transform = transform
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx, label, image_filename = self.data_list[idx]
        image = Image.open(image_filename).convert('RGB')
        image = image if self.transform is None else self.transform(image)
        return image, label, idx

    pass

