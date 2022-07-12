import torch
import torch.nn as nn
from .Normalize import Normalize


class LF2CSNet(nn.Module):

    def __init__(self, encoder, fsl_dim=1024, low_dim=1024):
        super().__init__()
        self.encoder = encoder
        self.fsl_dim = fsl_dim
        self.low_dim = low_dim

        self.fsl_header = nn.Linear(self.encoder.out_dim, self.fsl_dim)
        self.lf2cs_header = nn.Linear(self.encoder.out_dim, self.low_dim)
        self.l2norm = Normalize(2)
        pass

    def forward(self, x):
        out = self.encoder(x)

        fsl_logits = self.fsl_header(out)
        lf2cs_logits = self.lf2cs_header(out)
        lf2cs_l2norm = self.l2norm(lf2cs_logits)
        return fsl_logits, lf2cs_logits, lf2cs_l2norm

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass
