import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
from self_supervised.models.helper import get_encoder


class SimCLR(nn.Module):
    def __init__(self, img_size, backbone='resnet18'):
        super(SimCLR, self).__init__()
        self.f, projection_size = get_encoder(backbone, img_size)
        if img_size >= 100:
            projection_size = self.f.fc.out_features

        # projection head
        self.g = nn.Sequential(
                                nn.Linear(projection_size, 512, bias=False),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, 128, bias=True)
                               )

    def forward(self, x, y=None):
        x = self.f(x)
        feat_x = torch.flatten(x, start_dim=1)
        out_x = self.g(feat_x)

        if y is not None:
            y = self.f(y)
            feat_y = torch.flatten(y, start_dim=1)
            out_y = self.g(feat_y)
            return F.normalize(feat_x, dim=-1), F.normalize(feat_y, dim=-1), F.normalize(out_x, dim=-1),  F.normalize(out_y, dim=-1)
        else:
            return F.normalize(feat_x, dim=-1), F.normalize(out_x, dim=-1)


