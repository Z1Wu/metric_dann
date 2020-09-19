# from torchvision.datasets import MNIST
import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import Sampler
import collections
from torch.utils.data import DataLoader

class MNIST_base(torch.utils.data.Dataset):
    """`MNIST-M Dataset."""
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    # selected_class = [5, 6, 7, 8, 9]

    def __init__(self, root, train=True,
                 transform=None, 
                 target_transform=None,
                 download=False, 
                 label_filter = None):
        """Init MNIST-M dataset."""
        super(MNIST_base, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        
        self.type = self.root.split('/')[-1]

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.file = self.training_file
        else:
            self.file = self.test_file
        
        self.data, self.labels = torch.load(os.path.join(self.root,
                                self.processed_folder,
                                self.file))
        if not isinstance(self.data, torch.Tensor):
            self.data = torch.from_numpy(self.data)
            self.labels = torch.from_numpy(self.labels)
            
        # only choose specific class
        if label_filter:
            idxes =  []
            for idx in range(self.labels.shape[0]):
                if label_filter(self.labels[idx].item()):
                    idxes.append(idx)
            idxes = torch.Tensor(idxes).long()
            self.data = torch.index_select(self.data, 0, idxes)
            self.labels = torch.index_select(self.labels, 0, idxes)
        

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # if self.train:
        #     img, target = self.train_data[index], self.train_labels[index]
        # else:
        #     img, target = self.test_data[index], self.test_labels[index]
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(type(img))
        if self.type == 'MNIST':
            img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        elif self.type == 'MNIST-M':
            img = Image.fromarray(img.squeeze().numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           self.processed_folder,
                                           self.training_file)) and \
               os.path.exists(os.path.join(self.root,
                                           self.processed_folder,
                                           self.test_file))

class MPerClassSampler(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    paras:
    - labels[dataset_len, ] : all labels of the dataset, class number will get from this
    - m: number per class
    """
    def __init__(self, labels, m, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m_per_class = int(m)
        self.labels_to_indices = self.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class*len(self.labels)
        self.list_size = length_before_new_iter
        if self.length_of_single_pass < self.list_size:
            self.list_size -= (self.list_size) % (self.length_of_single_pass)

    def get_labels_to_indices(self, labels):
        """
        Creates labels_to_indices, which is a dictionary mapping each label
        to a numpy array of indices that will be used to index into self.dataset
        """
        labels_to_indices = collections.defaultdict(list)
        for i, label in enumerate(labels):
            labels_to_indices[label].append(i)
        for k, v in labels_to_indices.items():
            labels_to_indices[k] = np.array(v, dtype=np.int)
        return labels_to_indices
    
    def safe_random_choice(self, input_data, size):
        """
        Randomly samples without replacement from a sequence. It is "safe" because
        if len(input_data) < size, it will randomly sample WITH replacement
        Args:
            input_data is a sequence, like a torch tensor, numpy array,
                            python list, tuple etc
            size is the number of elements to randomly sample from input_data
        Returns:
            An array of size "size", randomly sampled from input_data
        """
        replace = len(input_data) < size
        return np.random.choice(input_data, size=size, replace=replace)

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0]*self.list_size
        i = 0
        num_iters = self.list_size // self.length_of_single_pass if self.length_of_single_pass < self.list_size else 1
        for _ in range(num_iters):
            np.random.shuffle(self.labels)
            for label in self.labels:
                t = self.labels_to_indices[label]
                idx_list[i:i+self.m_per_class] = self.safe_random_choice(t, size=self.m_per_class)
                i += self.m_per_class
        return iter(idx_list)

def to_random_rgb(x):
    color1 = np.random.randint(0, 256, size=3, dtype='uint8')
    color2 = np.random.randint(0, 256, size=3, dtype='uint8')
    x = np.array(x)
    x = x.astype('float32')/255.0
    x = np.expand_dims(x, 2)
    x = (1.0 - x) * color1 + x * color2
    return Image.fromarray(x.astype('uint8'))
    
def color_inverse(x:torch.Tensor):
    return 1 - x


def get_mnist_loader(dataset_root, train = True, label_filter = None): 
    """
    这里通过 resize 和 增加 channel 的方式，让 mnist 的图片尺寸变成 3 * 32 * 32
    """
    mean = (0.29730626, 0.29918741, 0.27534935)
    std = (0.32780124, 0.32292358, 0.32056796)
    if train:
        # TODO: 按照论文的说法这里需要使用 color intensity invers 以及 gray scale to rgb 的方法把这个 
        # mnist 的图片转化成 32 * 32 * 3 的图片。
        mnist_transform = transforms.Compose([
            # randomly color digit and background:
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(color_inverse),
            transforms.Normalize(mean, std)
        ])
        cls_num, sample_per_cls = 5, 2
        dataset = MNIST_base( # target domain
            root=dataset_root,
            train=train,
            transform=mnist_transform, 
            label_filter=label_filter
        )
        batch_sampler = MPerClassSampler(dataset.labels, 2)
        loader = DataLoader(
            dataset,
            batch_size= sample_per_cls * cls_num,
            sampler=batch_sampler,
            num_workers=2,
            drop_last=True
        )
        return loader
    else:
        pass

def get_mnist_m_loader(dataset_root, label_filter = None, train = True):
    mean = (0.29730626, 0.29918741, 0.27534935)
    std = (0.32780124, 0.32292358, 0.32056796)
    if train:
        mnistm_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std= std)
        ])
        dataset = MNIST_base( # source domain
            root=dataset_root,
            train=train,
            label_filter=label_filter,
            transform=mnistm_transform
        )
        cls_num, sample_per_cls = 5, 2
        batch_sampler = MPerClassSampler(dataset.labels, 2)
        loader = DataLoader(
            dataset, 
            batch_size= sample_per_cls * cls_num, 
            sampler=batch_sampler,
            num_workers=2,
            drop_last=True
        )
        return loader
    else:
        mnistm_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        dataset = MNIST_base( # source domain
            root=dataset_root,
            train=train,
            label_filter=label_filter,
            transform=mnistm_transform
        )
        loader = DataLoader(
            dataset, 
            batch_size= 256, 
            num_workers=2,
            drop_last=True
        )
        
def get_val_loader(dataset_root,  batch_size, label_filter = None,train = True):
    mean = (0.29730626, 0.29918741, 0.27534935)
    std = (0.32780124, 0.32292358, 0.32056796)
    mnistm_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std= std)
    ])
    dataset = MNIST_base( # source domain
        root=dataset_root,
        train=train,
        label_filter=label_filter,
        transform=mnistm_transform
    )
    loader = DataLoader(
        dataset, 
        batch_size= batch_size, 
        num_workers=2,
        drop_last=True
    )
    return loader

# TODO: 验证 mnistm 数据集的正确性
if __name__ == "__main__":
    root = './dataset'
    # trian_dataset = MNIST(root) # train
    # test_dataset = MNIST(root, train=False, download=True)
    # change form mnist to 
    