import torch
from torch import jit
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from torchvision.utils import make_grid
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

# revert imgs which have been normalized 
def img_visualization(
    img_tensor:torch.Tensor, 
    nrow:int, 
    mean:Tuple[float, float, float]= None, 
    std:Tuple[float, float, float] = None
    ):
    """ save image to writer
    Args:
    img_tensor[N, C, H, W]: 
    col_num: 
    mean:
    std:
    Return:
    img
    """
    # modify benchmark
    inverse_norm = None
    if mean != None: # inverse back the normalized image
        unnorm = NormalizeInverse(mean, std)
        img_tensor = torch.stack(list(map(unnorm, torch.unbind(img_tensor, 0))), 0)
    imgs = make_grid(img_tensor, nrow)
    # writer.add_image('img_vis', imgs)   
    return imgs 

def unnorm_imgs(imgs:torch.Tensor, mean, std):
    unnorm = NormalizeInverse(mean, std)
    img_tensor = torch.stack(list(map(unnorm, torch.unbind(imgs, 0))), 0)
    return img_tensor

def show_img_with_label(imgs:torch.Tensor, label:torch.Tensor, n_colum, mean = None, std = None):
    """
    input:
    - imgs[bn, c, h, w]
    - label[bn, ]
    - n_colum
    - mean
    - std
    output:
    - figure: matplotlib figure
    """
    bn, c, h, w = imgs.shape
    if mean:
        unnorm_imgs(imgs, mean, std)
    # image for show need to form chw to hwc
    imgs = imgs.permute([0, 2, 3, 1])
    assert imgs.shape == (bn, h, w, c)
    n_row = bn // n_colum
    if bn % n_colum != 0:
        n_row += 1
    print(n_row, n_colum, bn)
    fig = plt.figure(figsize=(n_colum * 2, n_row * 2)) # every img with
    if c == 3: # rgb image
        for idx in range(bn):
            ax = fig.add_subplot(n_row, n_colum, idx+1, xticks=[], yticks=[])
            ax.imshow(imgs[idx])
            ax.set_title(f'{label[idx].item()}')
    elif c == 1:
        for idx in range(bn):
            ax = fig.add_subplot(n_row, n_colum, idx+1, xticks=[], yticks=[])
            ax.imshow(imgs[idx].squeeze(), cmap = 'gray')
            ax.set_title(f'{label[idx].item()}')
    return fig

# def debug_info(feat:torch.Tensor, imgs:torch.Tensor, label:torch.Tensor, show_wrong_img = False):
#     """ output accurancy of current model, 
#     visualized the wrong predict pair.
#     input:
#     - feat[bn, feat_dim]
#     - imgs[bn, c, w, h]
#     - label[bn, ]
    
#     """
#     import sys
#     sys.path.append('.')
#     from lib.misc import binary_accurancy, generate_label_multi2bin
#     device = feat.device
#     lb = generate_label_multi2bin(label.unsqueeze(1)).to(device) # bn * bn, 1
#     y = F.sigmoid(feat.matmul(feat.t()).view(-1)).unsqueeze(-1) # bn * bn, 1 
#     acc, idxs = binary_accurancy(y, lb, need_idx=True)
#     if show_wrong_img:
#         img_num = imgs.shape[0]
#         wrong_img_pair = torch.stack((idxs.long() // img_num, idxs.long() % img_num), dim=1) # len, 2
#         imgs = imgs.index_select(0, wrong_img_pair.view(-1))
#         mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
#         res = img_visualization(imgs, 2, mean, std)
#         return acc, res
#     return acc
