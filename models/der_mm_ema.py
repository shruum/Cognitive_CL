# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy

import torch
from copy import deepcopy
from utils.buffer import Buffer
from torch.nn import functional as F
from utils.args import *
from utils.auxiliary import *
from models.utils.continual_model import ContinualModel
import torchvision.transforms as transforms

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' multi memory Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_auxiliary_args(parser)
    parser.add_argument('--alpha_mm', nargs='*', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--ema_alpha', type=float, required=True,
                        help='ema decay weight.')
    parser.add_argument('--ema_update_freq', type=float, required=True,
                        help='frequency.')
    return parser


class DerMMEma(ContinualModel):
    NAME = 'der_mm_ema'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform, aux_transform=None):
        super(DerMMEma, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.aux = AuxiliaryNet(self.args, self.device)
        self.net2 = deepcopy(backbone).to(self.device)
        self.ema_net2 = deepcopy(self.net2).to(self.device)
        self.opt2 = torch.optim.SGD(self.net2.parameters(), lr=self.args.lr)
        self.transform_aux = transforms.Compose(
            [transforms.ToPILImage(),
             transform_sobel_edge(self.args, self.args.shape_upsample_size),
             transforms.RandomCrop(self.args.img_size, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             #transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2615))
            ])
        self.global_step = 0
        self.current_task = 0

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        self.opt2.zero_grad()
        loss_dict = {}
        loss_aux_ema1 = loss_aux_ema2 = loss_buf1 = loss_buf2 = 0

        outputs1 = self.net(inputs)

        inputs_aux = deepcopy(not_aug_inputs)
        inputs_aux = self.aux.get_data(inputs_aux)
        outputs2 = self.net2(inputs_aux)

        loss_ce1 = self.loss(outputs1, labels)
        loss_ce2 = self.loss(outputs2, labels)

        loss1 = loss_ce1
        loss2 = loss_ce2

        if self.args.dir_aux:
            loss_aux12 = self.aux.loss(outputs1, outputs2.detach())
            loss_aux21 = self.aux.loss(outputs2, outputs1.detach())
            loss1 += (loss_aux12 * self.args.loss_wt[0])
            loss2 += (loss_aux21 * self.args.loss_wt[1])

        if not self.buffer.is_empty():
            buf_inputs1, buf_logits1, buf_logits2 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_inputs2 = self.buffer.get_data_aux(
                self.args.minibatch_size, transform=self.transform_aux)

            buf_outputs1 = self.net(buf_inputs1)
            buf_outputs2 = self.net2(buf_inputs2)
            buf_outputs_ema = self.ema_net2(buf_inputs2)

            loss_aux_ema1 = self.aux.loss(buf_outputs1, buf_outputs_ema.detach())
            loss_aux_ema2 = self.aux.loss(buf_outputs2, buf_outputs_ema.detach())

            loss_log_12 = F.mse_loss(buf_outputs1, buf_logits1)
            loss_log_21 = F.mse_loss(buf_outputs2, buf_logits2)
            loss1 += (loss_log_12 * self.args.alpha_mm[0]) + (loss_aux_ema1 * self.args.loss_wt[0])
            loss2 += (loss_log_21 * self.args.alpha_mm[1]) + (loss_aux_ema2 * self.args.loss_wt[1])

        self.aux.collate_loss(loss_dict, loss_ce=loss_ce1, loss_aux=loss_aux12, loss_aux_mem=loss_aux_ema1,
                              loss_logit_mem=loss_log_12, m1=True)
        self.aux.collate_loss(loss_dict, loss_ce=loss_ce2, loss_aux=loss_aux21, loss_aux_mem=loss_aux_ema2,
                              loss_logit_mem=loss_log_21, m1=False)

        if hasattr(self, 'writer'):
            for loss_name, loss_item in loss_dict.items():
                self.writer.add_scalar('Task {}/{}'.format(self.current_task, loss_name), loss_item,
                                       global_step=self.iteration)

        loss1.backward()
        loss2.backward()
        self.opt.step()
        self.opt2.step()

        self.buffer.add_data(examples=not_aug_inputs, logits=outputs1.data, logits_aux=outputs2.data)

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.args.ema_update_freq:
            self.update_ema_model_variables()

        return loss1.item()+loss2.item()


    def update_ema_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.args.ema_alpha)
        for ema_param, param in zip(self.ema_net2.parameters(), self.net2.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def end_task(self, dataset) -> None:
        self.current_task += 1
