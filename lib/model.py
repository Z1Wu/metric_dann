import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeatureExtractor(nn.Module):
    """
    input:
    - X[bn, 32, 32, 3]: input image
    output:
    - embeddings[bn, 128]: 
    """
    def __init__(self) -> None:
        super(FeatureExtractor, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(32, 32, 3, padding=1)

        self.maxpool = torch.nn.MaxPool2d(2)

        self.conv2_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(64, 64, 3, padding=1)

        self.conv3_1 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(128, 128, 3, padding=1)

        self.fc1 = torch.nn.Linear(4*4*128, 128)
        self.fc2 = torch.nn.Linear(128,128)
        
        self.scale_factor = 2 ** 0.5

        for m in self.modules():  # parameter 
            if isinstance(m, nn.Conv2d):
                # TODO: init module, should output name and 
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        

    def forward(self, X:torch.Tensor):
        out = F.relu(self.conv1_1(X))
        out = F.relu(self.conv1_2(out))
        out = self.maxpool(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.maxpool(out)

        out = F.relu(self.conv3_1(out))
        out	= F.relu(self.conv3_2(out))
        out = self.maxpool(out)

        out = torch.flatten(out, start_dim=1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = F.normalize(out, 2) * self.scale_factor # norm and scale

        return out