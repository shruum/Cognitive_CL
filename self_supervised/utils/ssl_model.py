import torch.nn as nn
from torch.optim import Adam
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from self_supervised.augmentations import SimCLRTransform
from self_supervised.models import SimCLR
from self_supervised.criterion import NTXent
from utils.buffer import Buffer


class ContinualSSLModel(nn.Module):
    """
    Continual self-supervised learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, args: Namespace, img_size=32, backbone='simclr', ) -> None:
        super(ContinualSSLModel, self).__init__()
        if backbone == 'simclr':
            self.net = SimCLR(img_size)
        self.loss = NTXent()
        self.args = args
        self.transform = SimCLRTransform(size=img_size)
        self.opt = Adam(self.net.parameters(), lr=3e-4)
        self.device = get_device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def observe(self, inputs: torch.Tensor, buffer: Buffer = None) -> float:
        self.opt.zero_grad()

        if buffer and not buffer.is_empty():
            buf_inputs, buf_labels = buffer.get_data(self.args.minibatch_size)
            inputs = torch.cat((inputs, buf_inputs))

        x, y = self.transform(inputs)

        _, _, zx, zy = self.net(x, y)
        loss = self.loss(zx, zy)
        loss.backward()
        self.opt.step()

        return loss.item()
