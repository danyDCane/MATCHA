import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init # 補上這行避免 conv_init 報錯
from torch.autograd import Variable
import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)

def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, depth, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, return_blocks: bool = False):
        """
        Forward pass.

        Args:
            x: input tensor [B, 3, H, W]
            return_blocks: if True, also return outputs of the first three
                           convolutional blocks (layer1, layer2, layer3).

        Returns:
            If return_blocks is False (default):
                logits: [B, num_classes]
            If return_blocks is True:
                logits: [B, num_classes]
                features: dict with keys 'layer1', 'layer2', 'layer3',
                          each of shape [B, C, H, W].
        """
        out = F.relu(self.bn1(self.conv1(x)))

        # First three convolutional blocks
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        # Use adaptive average pooling to support different input sizes
        # (e.g., 32x32 for CIFAR-10, 224x224 for PACS)
        feat = F.adaptive_avg_pool2d(out3, 1)
        feat = feat.view(feat.size(0), -1)
        logits = self.linear(feat)

        if return_blocks:
            features = {
                "layer1": out1,
                "layer2": out2,
                "layer3": out3,
            }
            return logits, features

        return logits

class StandardResNetWrapper(nn.Module):
    """
    Wrapper for torchvision ResNet to support return_blocks functionality.
    This allows extracting intermediate features from layer1, layer2, layer3.
    """
    def __init__(self, depth, num_classes):
        super(StandardResNetWrapper, self).__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
        
        # 根據 depth 選擇對應的 ResNet
        resnet_dict = {
            18: resnet18,
            34: resnet34,
            50: resnet50,
            101: resnet101,
            152: resnet152
        }
        
        if depth not in resnet_dict:
            raise ValueError(f"ResNet depth {depth} not supported. Choose from {list(resnet_dict.keys())}")
        
        # 創建 ResNet 模型（pretrained=False）
        self.backbone = resnet_dict[depth](pretrained=False)
        
        # 修改最後一層以匹配 num_classes
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x, return_blocks: bool = False):
        """
        Forward pass with optional intermediate feature extraction.
        
        Args:
            x: input tensor [B, 3, H, W]
            return_blocks: if True, also return outputs of layer1, layer2, layer3
        
        Returns:
            If return_blocks is False:
                logits: [B, num_classes]
            If return_blocks is True:
                logits: [B, num_classes]
                features: dict with keys 'layer1', 'layer2', 'layer3'
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # Extract intermediate features if needed
        if return_blocks:
            out1 = self.backbone.layer1(x)
            out2 = self.backbone.layer2(out1)
            out3 = self.backbone.layer3(out2)
            out4 = self.backbone.layer4(out3)
            
            # Global average pooling
            feat = self.backbone.avgpool(out4)
            feat = feat.view(feat.size(0), -1)
            logits = self.backbone.fc(feat)
            
            features = {
                "layer1": out1,
                "layer2": out2,
                "layer3": out3,
            }
            return logits, features
        else:
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            x = x.view(x.size(0), -1)
            logits = self.backbone.fc(x)
            return logits

if __name__ == '__main__':
    net=ResNet(50, 10)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
