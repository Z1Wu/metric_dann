import logging
import torch
import sys
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data.dataloader import DataLoader
from typing import List, Tuple
import argparse

def binary_accurancy(pred, label, need_idx = False):
    """
    parameter:
    - pred[n, 1]: sigmoid output, within the scope of [0, 1]
    - label[n, 1]: 0 means diff, 1 means same
    returns:
    - t_acc[] : 
    - idxs(optional)[len_of_wrong]
    """
    preds = (pred > 0.5).float()
    assert preds.shape == label.shape
    t_acc = (preds == label).float().mean()
    if need_idx:
        return t_acc, torch.nonzero((preds != label), as_tuple=True)[0]
    return t_acc

from typing import Optional
from torch.optim.optimizer import Optimizer

class StepwiseLR:
    """
    A lr_scheduler that update learning rate using the following schedule:

    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

    where `i` is the iteration steps.

    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1

class ForeverDataIterator:
    """A data iterator that will never stop producing data"""
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)