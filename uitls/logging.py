import logging
import torch
import sys
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data.dataloader import DataLoader
from typing import List, Tuple
import argparse

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name=None, fmt=':f', tb_tag=None, writer:SummaryWriter = None):
        self.name = name
        self.fmt = fmt
        self.tb_tag = tb_tag
        self.reset()

        # attach to writer
        self.writer = writer

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, iter = None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.writer and iter != None: # non-none iter means save the record val into writer
            self.writer.add_scalar(self.tb_tag, val, iter)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger = None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.logger == None:
            print('\t'.join(entries))
        else:
            self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def update_meter_n_witer(recorder:AverageMeter, val, num, iter, writer):
    # update 
    recorder.update(val, num)
    writer.add_scalar(recorder.tb_tag, val, iter)

def experiment_info(logger:logging.Logger, models:List[torch.nn.Module], datalaoders:List[Tuple[str, DataLoader]], args:argparse.Namespace, exp_info:str):
    """ record experiment information into logger file
    """
    logger.info('=' * 10 + 'exp info')
    logger.info(exp_info)

    logger.info('=' * 10 + 'args info')
    str_arg = '\n'.join([str(e) for e in args._get_kwargs()])
    logger.info(str_arg)

    logger.info('=' * 10 + 'model info( train_params )')
    str_model = ''
    for model in models:
        str_model += '\n'.join([n for n,p in model.named_parameters() if p.requires_grad == True])
    logger.info(str_model)

    logger.info('=' * 10 + 'datset info')   
    str_ds = '' 
    for name, loader in datalaoders:
        str_ds += f'\nlen of {name}: {len(loader.dataset)}'
    logger.info(str_ds)

def get_current_time(time_format) -> str:
    from datetime import datetime
    from dateutil import tz
    return datetime.now(tz=tz.gettz('Asia/Shanghai')).strftime(time_format)

def configure_logger(logger_name, log_dir):
    # Create a custom logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # remove all default handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    from datetime import datetime
    current_time = get_current_time('%m%d%H%M')
    output_file = f'{os.path.join(log_dir, current_time)}.log'
    f_handler = logging.FileHandler(output_file, mode='w') # recreate file
    c_handler.setLevel(logging.DEBUG) # message above args will be log
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.debug(f"created logger {logger_name}, write logger file to {output_file}")

    return logger


def generate_tensor_board_name(bn, lr, runner_name='', base_dir='runs', comment='', args = None):
    # bn: batch_size, lr: learning_rate
    if args != None and (args.mode == 'debug' or args.mode == 'test'):
        base_dir = 'debug_' + base_dir
    current_time = get_current_time('%m%d%H%M')
    log_dir = os.path.join(
        base_dir, f'{runner_name}_{current_time}_b_{bn}_l_{lr}_{comment}')
    return log_dir

def config_tensor_board_writer(bn, lr, logger, runner_name='', base_dir='runs', comment='', args = None):
    # bn: batch_size, lr: learning_rate
    if args != None and (args.mode == 'debug' or args.mode == 'test'):
        base_dir = 'debug_' + base_dir
    current_time = get_current_time('%m%d%H%M')
    log_dir = os.path.join(
        base_dir, f'{runner_name}_{current_time}_b_{bn}_l_{lr}_{comment}')
    logger.info(f'save tensorboard log to {log_dir}')
    return SummaryWriter(log_dir=log_dir)


def show_embedding(backbone, data_loader_list, tag, epoch, writer, device):
    backbone.eval()
    with torch.no_grad():
        feature:torch.Tensor = torch.Tensor()
        tot_label:torch.Tensor = torch.Tensor().long()
        for data_loader in data_loader_list:
            for img, label in data_loader:
                img = img.to(device)
                feat = backbone(img)
                feature = torch.cat([feature, feat.cpu()], 0)
                tot_label = torch.cat([tot_label, label], 0)
        writer.add_embedding(feature, tot_label, tag = tag, global_step=epoch)