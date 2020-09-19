from lib.model import FeatureExtractor
from dalib.adaptation import DomainAdversarialLoss
from utils.misc import ForeverDataIterator
from utils.logging import AverageMeter, ProgressMeter
from pytorch_metric_learning import NPairsLoss
from lib.evaluation import NMI_eval
from torch.optim import Adam
import itertools
import torch
import os

def source_only_train(feature_extractor:FeatureExtractor, 
    domain_adv:DomainAdversarialLoss, 
    metric_loss:torch.nn.Module,
    src_iter:ForeverDataIterator, 
    tar_iter:ForeverDataIterator, 
    src_val_loader, tar_val_loader,
    args):
    optimizer = Adam(
        itertools.chain(feature_extractor.parameters(), domain_adv.parameters()),
        lr= args.lr,weight_decay=args.weight_decay
    )
    
    # misc
    logger = args.logger
    device = args.device
    model_dir = args.model_dir
    
    # loss
    loss_rec = AverageMeter('tot_loss', tb_tag='Loss/tot', writer=args.writer)
    loss_lb_rec = AverageMeter('lb_loss', tb_tag='Loss/lb', writer=args.writer)
    loss_lb_g_rec = AverageMeter('lb_g_loss', tb_tag='Loss/lb_g', writer=args.writer)

    n_iter = 0
    best_nmi = 0
    for e_i in range(args.epoch):
        feature_extractor.train()
        domain_adv.train()
        progress = ProgressMeter(
            args.iter_per_epoch,
            [loss_lb_g_rec, loss_lb_rec],
            prefix="Epoch: [{}]".format(e_i),
            logger=args.logger
        )
        for i in range(args.iter_per_epoch):
            x_s, l_s = next(src_iter)
            x_t, l_t = next(tar_iter)
            # for obj in [x_s, x_t, l_s, l_t]: # to device
                # obj = obj.to(device)
            
            x_s, l_s, x_t, l_t = x_s.to(device), l_s.to(device), x_t.to(device), l_t.to(device)

            x = torch.cat((x_s, x_t), dim=0)
            f, g = feature_extractor(x)
            f_s, f_t = f.chunk(2, dim=0)
            g_s, g_t = g.chunk(2, dim=0)
            
            # source only part
            loss_s = metric_loss(f_s, l_s) # get n-pair loss on source domain
            loss_s_g = metric_loss(g_s, l_s) # get n-pair loss on source domain
            loss_lb_rec.update(loss_s.item(), x_s.size(0), iter=n_iter)
            loss_lb_g_rec.update(loss_s_g.item(), x_s.size(0), iter=n_iter)
            
            loss = 0.5 * (loss_s + loss_s_g)
            # loss = loss_s
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_iter += 1
            if i % args.print_freq == 0:
                progress.display(i)

        if e_i % 5 == 0:
            # global_logger.info(f"saving embedding in epoch{e_i}")
            # # show embedding
            # show_embedding(backbone, [src_val_loader], tag=f'src_{e_i}', epoch=e_i, writer, device)
            # show_embedding(backbone, [tar_val_loader], tag=f'tar_{e_i}', epoch=e_i, writer, device)
            
            nmi = NMI_eval(feature_extractor, src_val_loader, 5, device, type='src')
            logger.info(f'test on train set nmi: {nmi}')
            nmi = NMI_eval(feature_extractor, tar_val_loader, 5, device, type='tar')
            logger.info(f'test on test set nmi: {nmi}')
            if nmi > best_nmi:
                logger.info(f"save best model to {model_dir}")
                torch.save(feature_extractor.state_dict(), os.path.join(model_dir, 'minst_best_model.pth'))
                best_nmi = nmi

def dann_train(feature_extractor:FeatureExtractor, 
    domain_adv:DomainAdversarialLoss, 
    src_iter:ForeverDataIterator, 
    tar_iter:ForeverDataIterator, 
    src_val_loader, tar_val_loader,
    args):
    optimizer = Adam(
        itertools.chain(feature_extractor.parameters(), domain_adv.parameters()),
        lr= args.lr,weight_decay=args.weight_decay
    )
    
    npair_loss = NPairsLoss()  # n pair loss

    epoch = args.epoch
    iter_per_epoch = args.iter_per_epoch
    writer = args.writer # Summary Writer
    logger = args.logger
    device = args.device
    w_da = args.w_da
    model_dir = args.model_dir

    # loss
    loss_rec = AverageMeter('tot_loss', tb_tag='Loss/tot', writer=writer)
    loss_lb_rec = AverageMeter('lb_loss', tb_tag='Loss/lb', writer=writer)
    loss_lb_g_rec = AverageMeter('lb_g_loss', tb_tag='Loss/lb_g', writer=writer)
    loss_da_rec = AverageMeter('da_loss', tb_tag='Loss/da', writer=writer)

    # acc
    da_acc_rec = AverageMeter('da_acc', tb_tag='Acc/da', writer=writer)

    n_iter = 0
    best_nmi = 0
    for e_i in range(epoch):
        feature_extractor.train()
        domain_adv.train()
        progress = ProgressMeter(
            iter_per_epoch,
            [loss_lb_g_rec, loss_lb_rec, loss_da_rec,da_acc_rec],
            prefix="Epoch: [{}]".format(e_i),
            logger=logger
        )
        for i in range(iter_per_epoch):
            x_s, l_s = next(src_iter)
            x_t, l_t = next(tar_iter)
            # for obj in [x_s, x_t, l_s, l_t]: # to device
                # obj = obj.to(device)
            
            x_s, l_s, x_t, l_t = x_s.to(device), l_s.to(device), x_t.to(device), l_t.to(device)

            x = torch.cat((x_s, x_t), dim=0)
            f, g = feature_extractor(x)
            f_s, f_t = f.chunk(2, dim=0)
            g_s, g_t = g.chunk(2, dim=0)
            
            # source only part
            loss_s = npair_loss(f_s, l_s) # get n-pair loss on source domain
            loss_s_g = npair_loss(g_s, l_s) # get n-pair loss on source domain
            loss_lb_rec.update(loss_s.item(), x_s.size(0), iter=n_iter)
            loss_lb_g_rec.update(loss_s_g.item(), x_s.size(0), iter=n_iter)
            
            # dann
            # da_loss = domain_adv(f_s,f_t)
            da_loss = domain_adv(g_s,f_t)
            domain_acc = domain_adv.domain_discriminator_accuracy
            loss_da_rec.update(da_loss.item(), f.size(0), iter=n_iter)
            da_acc_rec.update(domain_acc.item(), f.size(0), iter=n_iter)

            loss = 0.5 * (loss_s + loss_s_g) + w_da * da_loss
            # loss = loss_s
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_iter += 1
            if i % print_freq == 0:
                progress.display(i)

        if e_i % 5 == 0:
            # logger.info(f"saving embedding in epoch{e_i}")
            # # show embedding
            # show_embedding(backbone, [src_val_loader], tag=f'src_{e_i}', epoch=e_i, writer, device)
            # show_embedding(backbone, [tar_val_loader], tag=f'tar_{e_i}', epoch=e_i, writer, device)
            
            nmi = NMI_eval(feature_extractor, src_val_loader, 5, device, type='src')
            logger.info(f'test on train set nmi: {nmi}')
            nmi = NMI_eval(feature_extractor, tar_val_loader, 5, device, type='tar')
            logger.info(f'test on test set nmi: {nmi}')
            if nmi > best_nmi:
                logger.info(f"save best model to {model_dir}")
                torch.save(feature_extractor.state_dict(), os.path.join(model_dir, 'minst_best_model.pth'))
                best_nmi = nmi

