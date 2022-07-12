import torch
import torch.nn as nn
from .Normalize import Normalize


class C4Net(nn.Module):

    def __init__(self, hid_dim, z_dim, has_norm=False):
        super().__init__()
        self.conv_block_1 = nn.Sequential(nn.Conv2d(3, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 41
        self.conv_block_2 = nn.Sequential(nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 21
        self.conv_block_3 = nn.Sequential(nn.Conv2d(hid_dim, hid_dim, 3, padding=1),
                                          nn.BatchNorm2d(hid_dim), nn.ReLU(), nn.MaxPool2d(2))  # 10
        self.conv_block_4 = nn.Sequential(nn.Conv2d(hid_dim, z_dim, 3, padding=1),
                                          nn.BatchNorm2d(z_dim), nn.ReLU(), nn.MaxPool2d(2))  # 5

        self.has_norm = has_norm
        if self.has_norm:
            self.norm = Normalize(2)
        pass

    def forward(self, x):
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)

        if self.has_norm:
            out = out.view(out.shape[0], -1)
            out = self.norm(out)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class EncoderC4(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = C4Net(64, 64)
        self.out_dim = 64
        pass

    def forward(self, x):
        out = self.encoder(x)
        out = torch.flatten(out, 1)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass
