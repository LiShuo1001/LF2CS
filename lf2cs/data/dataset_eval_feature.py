from PIL import Image
from torch.utils.data import Dataset


class EvalFeatureDataset(Dataset):

    def __init__(self, data_list, transform):
        self.data_list = data_list
        self.transform = transform
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_filename = self.data_list[idx][-1]
        image = Image.open(image_filename).convert('RGB')
        image = self.transform(image)
        return image, image_filename

    pass

