import torch
from torch.utils.tensorboard import SummaryWriter
from lib.model import FeatureExtractor
# from uitls.logging import generate_tensor_board_name
import os
from lib.mnist_loader import get_val_loader
from uitls.logging import show_embedding


exp_root = 'exp/mnist'
model_name = 'minst_backbone_09171458'
writer = SummaryWriter(log_dir=f'runs/test_{model_name}_source_only')
device = torch.device("cuda")

if __name__ == '__main__':
    
    model_path = os.path.join(exp_root, model_name+'.pth')
    # load the model
    model = FeatureExtractor().to(device)
    
    print(f'load model {model_path}')
    model.load_state_dict(torch.load(model_path))
    
    src_eval_loader = get_val_loader(
        dataset_root = './dataset/MNIST-M', 
        train=True, 
        label_filter=lambda x : x in [0, 1, 2, 3, 4],
        batch_size=256
    )
    tar_eval_loader = get_val_loader(
        dataset_root = './dataset/MNIST', 
        train=True,
        label_filter=lambda x : x in [5, 6, 7, 8, 9],
        batch_size=256
    )

    show_embedding(model, [src_eval_loader], 'test_src', 0, writer, device)
    show_embedding(model, [tar_eval_loader], 'test_tar', 0, writer, device)
    