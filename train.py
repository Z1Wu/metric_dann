from os import write
from lib.model import FeatureExtractor
from lib.mnist_loader import get_mnist_loader, get_mnist_m_loader, get_val_loader
from uitls.misc import ForeverDataIterator
from uitls.logging import config_tensor_board_writer, configure_logger, AverageMeter, \
    ProgressMeter, get_current_time, generate_tensor_board_name, show_embedding
from dalib.adaptation.dann import DomainAdversarialLoss
from dalib.modules.domain_discriminator import DomainDiscriminator
from torch.optim import Adam
import torch
import os
import itertools
from pytorch_metric_learning.losses.n_pairs_loss import NPairsLoss
from torch.utils.tensorboard import SummaryWriter

cls_num, sample_num_per_cls = 5, 2
exp_root = './exp'
bn_size = cls_num * sample_num_per_cls
lr = 0.0003
weight_decay = 0.0001
epoch = 30
iter_per_epoch = 3000 # every epoch will iter 
print_freq = 500
device = torch.device("cuda")

# writer = None
# global_logger = None # global logger, for convenience
# setup up misc
runner_name = 'mnist'
model_dir = os.path.join(exp_root, runner_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
global_logger= configure_logger(runner_name, model_dir)
writer_log_dir = generate_tensor_board_name(bn = bn_size, lr = lr, runner_name=runner_name)
global_logger.info(f'save tensorboard log to {writer_log_dir}')
writer = SummaryWriter(log_dir=writer_log_dir)
exp_time = get_current_time('%m%d%H%M')

w_da = None
w_fea_recon_src = None
w_fea_recon_tar = None

def train(feature_extractor:FeatureExtractor, domain_adv:DomainAdversarialLoss, src_iter:ForeverDataIterator, tar_iter:ForeverDataIterator, src_val_loader, tar_val_loader):
    optimizer = Adam(
        itertools.chain(feature_extractor.parameters(), domain_adv.parameters()),
        lr= lr,weight_decay=weight_decay
    )
    
    npair_loss = NPairsLoss()  # n pair loss 

    # loss
    loss_rec = AverageMeter('tot_loss', tb_tag='Loss/tot', writer=writer)
    loss_lb_rec = AverageMeter('lb_loss', tb_tag='Loss/lb', writer=writer)
    # loss_ulb_rec = AverageMeter('ulb_loss', tb_tag='Loss/ulb')
    loss_da_rec = AverageMeter('da_loss', tb_tag='Loss/da', writer=writer)
    # acc
    da_acc_rec = AverageMeter('da_acc', tb_tag='Acc/da', writer=writer)

    n_iter = 0
    for e_i in range(epoch):
        feature_extractor.train()
        domain_adv.train()
        progress = ProgressMeter(
            iter_per_epoch,
            [loss_rec, loss_lb_rec, loss_da_rec,da_acc_rec],
            prefix="Epoch: [{}]".format(e_i),
            logger=global_logger
        )
        for i in range(iter_per_epoch):
            x_s, l_s = next(src_iter)
            x_t, l_t = next(tar_iter)
            # for obj in [x_s, x_t, l_s, l_t]: # to device
            #     obj = obj.to(device)
            
            x_s, l_s, x_t, l_t = x_s.to(device), l_s.to(device), x_t.to(device), l_t.to(device)

            x = torch.cat((x_s, x_t), dim=0)
            f = feature_extractor(x)
            f_s, f_t = f.chunk(2, dim=0)
            
            # source only part
            loss_s = npair_loss(f_s, l_s) # get n-pair loss on source domain
            loss_lb_rec.update(loss_s.item(), x_s.size(0), iter=n_iter)
            
            # dann
            da_loss = domain_adv(f_s,f_t)
            domain_acc = domain_adv.domain_discriminator_accuracy
            loss_da_rec.update(da_loss.item(), f.size(0), iter=n_iter)
            da_acc_rec.update(domain_acc.item(), f.size(0), iter=n_iter)

            loss = loss_s + da_loss
            # loss = loss_s
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_iter += 1
            if i % print_freq == 0:
                progress.display(i)

        # if e_i % 5 == 0:
        #     global_logger.info(f"saving embedding in epoch{e_i}")
        #     # show embedding
        #     show_embedding(backbone, [src_val_loader], tag=f'src_{e_i}', epoch=e_i, writer, device)
        #     show_embedding(backbone, [tar_val_loader], tag=f'tar_{e_i}', epoch=e_i, writer, device)

# def test()

if __name__ == "__main__":
    # setup model
    backbone = FeatureExtractor().to(device)
    domain_discri = DomainDiscriminator(in_feature=128, hidden_size=256).to(device)
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)
    
    # TODO feature reconstruction loss
    # TODO feautre transfer module

    src_domain_class = [0, 1, 2, 3, 4]
    tar_domain_class = [5, 6, 7, 8, 9]
    # setup dataloader
    src_train_loader = get_mnist_m_loader(
        dataset_root = './dataset/MNIST-M', 
        train=True, 
        label_filter=lambda x : x in src_domain_class
    )
    tar_train_loader = get_mnist_loader(
        dataset_root = './dataset/MNIST', 
        train=True,
        label_filter=lambda x : x in tar_domain_class
    )
    src_iter, tar_iter = ForeverDataIterator(src_train_loader), ForeverDataIterator(tar_train_loader)
    
    src_eval_loader = get_val_loader(
        dataset_root = './dataset/MNIST-M', 
        train=True, 
        label_filter=lambda x : x in src_domain_class,
        batch_size=256
    )

    tar_eval_loader = get_val_loader(
        dataset_root = './dataset/MNIST', 
        train=True,
        label_filter=lambda x : x in tar_domain_class,
        batch_size=256
    )
    
    # show_embedding(backbone, [tar_eval_loader], 'src', 0)

    train(backbone, domain_adv, src_iter, tar_iter, src_eval_loader, tar_eval_loader)
    show_embedding(backbone, [src_eval_loader], 'src', 0, writer, device)
    show_embedding(backbone, [tar_eval_loader], 'tar', 0, writer, device)
    # save model into file
    saved_model_path = os.path.join(model_dir, f'minst_backbone_{exp_time}.pth') 
    torch.save(backbone.state_dict(), saved_model_path)
    global_logger.info("model saved to {}.".format(saved_model_path))
    writer.close()
    