import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init # 補上這行避免 conv_init 報錯
from torch.autograd import Variable
import sys
import numpy as np
from style_transforms import StyleShift, StyleExplore, MixStyle
import random
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
    def __init__(self, depth, num_classes, use_style_shift=False, style_shift_prob=0.5, style_shift_ratio=0.5, 
                 style_explore_alpha=3.0, style_explore_ratio=0.5, mixstyle_alpha=0.1):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.style_shift_prob = style_shift_prob  # 保存为实例属性
        self.style_shift_ratio = style_shift_ratio  # 也可以保存（如果需要）
        self.use_style_shift = use_style_shift
        self.style_explore_alpha = style_explore_alpha
        self.style_explore_ratio = style_explore_ratio
        self.mixstyle_alpha = mixstyle_alpha

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)
        
        # Initialize StyleShift, StyleExplore, and MixStyle modules for each layer
        if self.use_style_shift:
            self.style_shift1 = StyleShift(activation_prob=style_shift_prob, shift_ratio=style_shift_ratio)
            self.style_shift2 = StyleShift(activation_prob=style_shift_prob, shift_ratio=style_shift_ratio)
            self.style_shift3 = StyleShift(activation_prob=style_shift_prob, shift_ratio=style_shift_ratio)
            # StyleExplore unconditionally follows StyleShift
            self.style_explore1 = StyleExplore(alpha=style_explore_alpha, explore_ratio=style_explore_ratio)
            self.style_explore2 = StyleExplore(alpha=style_explore_alpha, explore_ratio=style_explore_ratio)
            self.style_explore3 = StyleExplore(alpha=style_explore_alpha, explore_ratio=style_explore_ratio)
            # MixStyle follows StyleExplore
            self.mixstyle1 = MixStyle(alpha=mixstyle_alpha)
            self.mixstyle2 = MixStyle(alpha=mixstyle_alpha)
            self.mixstyle3 = MixStyle(alpha=mixstyle_alpha)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def extract_features_to_layer3(self, x):
        """
        Extract features up to layer3 without applying style shift.
        Used for style statistics computation in the first forward pass.
        
        Args:
            x: input tensor [B, 3, H, W]
        
        Returns:
            features: dict with keys 'layer1', 'layer2', 'layer3',
                     each of shape [B, C, H, W]
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        
        features = {
            "layer1": out1,
            "layer2": out2,
            "layer3": out3,
        }
        return features

    def forward(self, x, return_blocks: bool = False, communicator=None, debug_style_shift=False, iter_num=-1, rank=-1):
        """
        Forward pass.

        Args:
            x: input tensor [B, 3, H, W]
            return_blocks: if True, also return outputs of the first three
                           convolutional blocks (layer1, layer2, layer3).
            communicator: Communicator object with neighbor_style_stats attribute
                          (used for style shift if enabled)
            debug_style_shift: If True, print debug information for style shift
            iter_num: Current iteration number (for debugging)
            rank: Current rank (for debugging)

        Returns:
            If return_blocks is False (default):
                logits: [B, num_classes]
            If return_blocks is True:
                logits: [B, num_classes]
                features: dict with keys 'layer1', 'layer2', 'layer3',
                          each of shape [B, C, H, W].
        """
        out = F.relu(self.bn1(self.conv1(x)))

        # First three convolutional blocks with optional style shift
        out1 = self.layer1(out)
        if self.use_style_shift and communicator is not None:
            if self.training and random.random() <= self.style_shift_prob:
                out1 = self.style_shift1(out1, "layer1", communicator, self.training,
                                        verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # StyleExplore unconditionally follows StyleShift
                out1 = self.style_explore1(out1, "layer1", self.training,
                                          verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # MixStyle follows StyleExplore
                out1 = self.mixstyle1(out1, "layer1", self.training,
                                     verbose=debug_style_shift, iter_num=iter_num, rank=rank)
        
        out2 = self.layer2(out1)
        if self.use_style_shift and communicator is not None:
            if self.training and random.random() <= self.style_shift_prob:
                out2 = self.style_shift2(out2, "layer2", communicator, self.training,
                                        verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # StyleExplore unconditionally follows StyleShift
                out2 = self.style_explore2(out2, "layer2", self.training,
                                          verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # MixStyle follows StyleExplore
                out2 = self.mixstyle2(out2, "layer2", self.training,
                                     verbose=debug_style_shift, iter_num=iter_num, rank=rank)
        
        out3 = self.layer3(out2)
        if self.use_style_shift and communicator is not None:
            if self.training and random.random() <= self.style_shift_prob:
                out3 = self.style_shift3(out3, "layer3", communicator, self.training,
                                        verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # StyleExplore unconditionally follows StyleShift
                out3 = self.style_explore3(out3, "layer3", self.training,
                                          verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # MixStyle follows StyleExplore
                out3 = self.mixstyle3(out3, "layer3", self.training,
                                     verbose=debug_style_shift, iter_num=iter_num, rank=rank)

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
    def __init__(self, depth, num_classes, use_style_shift=False, style_shift_prob=0.5, style_shift_ratio=0.5,
                 style_explore_alpha=3.0, style_explore_ratio=0.5, mixstyle_alpha=0.1):
        super(StandardResNetWrapper, self).__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
        self.style_shift_prob = style_shift_prob  # 保存为实例属性
        self.style_shift_ratio = style_shift_ratio  # 也可以保存（如果需要）
        self.use_style_shift = use_style_shift
        self.style_explore_alpha = style_explore_alpha
        self.style_explore_ratio = style_explore_ratio
        self.mixstyle_alpha = mixstyle_alpha
        
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
        
        # Initialize StyleShift, StyleExplore, and MixStyle modules for each layer
        if self.use_style_shift:
            self.style_shift1 = StyleShift(activation_prob=style_shift_prob, shift_ratio=style_shift_ratio)
            self.style_shift2 = StyleShift(activation_prob=style_shift_prob, shift_ratio=style_shift_ratio)
            self.style_shift3 = StyleShift(activation_prob=style_shift_prob, shift_ratio=style_shift_ratio)
            # StyleExplore unconditionally follows StyleShift
            self.style_explore1 = StyleExplore(alpha=style_explore_alpha, explore_ratio=style_explore_ratio)
            self.style_explore2 = StyleExplore(alpha=style_explore_alpha, explore_ratio=style_explore_ratio)
            self.style_explore3 = StyleExplore(alpha=style_explore_alpha, explore_ratio=style_explore_ratio)
            # MixStyle follows StyleExplore
            self.mixstyle1 = MixStyle(alpha=mixstyle_alpha)
            self.mixstyle2 = MixStyle(alpha=mixstyle_alpha)
            self.mixstyle3 = MixStyle(alpha=mixstyle_alpha)
    
    def extract_features_to_layer3(self, x):
        """
        Extract features up to layer3 without applying style shift.
        Used for style statistics computation in the first forward pass.
        
        Args:
            x: input tensor [B, 3, H, W]
        
        Returns:
            features: dict with keys 'layer1', 'layer2', 'layer3',
                     each of shape [B, C, H, W]
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        out1 = self.backbone.layer1(x)
        out2 = self.backbone.layer2(out1)
        out3 = self.backbone.layer3(out2)
        
        features = {
            "layer1": out1,
            "layer2": out2,
            "layer3": out3,
        }
        return features
    
    def forward(self, x, return_blocks: bool = False, communicator=None, debug_style_shift=False, iter_num=-1, rank=-1):
        """
        Forward pass with optional intermediate feature extraction.
        
        Args:
            x: input tensor [B, 3, H, W]
            return_blocks: if True, also return outputs of layer1, layer2, layer3
            communicator: Communicator object with neighbor_style_stats attribute
                          (used for style shift if enabled)
            debug_style_shift: If True, print debug information for style shift
            iter_num: Current iteration number (for debugging)
            rank: Current rank (for debugging)
        
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
            if self.use_style_shift and communicator is not None:
                if self.training and random.random() <= self.style_shift_prob:
                    out1 = self.style_shift1(out1, "layer1", communicator, self.training, 
                                            verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                    # StyleExplore unconditionally follows StyleShift
                    out1 = self.style_explore1(out1, "layer1", self.training,
                                              verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                    # MixStyle follows StyleExplore
                    out1 = self.mixstyle1(out1, "layer1", self.training,
                                         verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                elif debug_style_shift:
                    print(f"[StyleShift layer1] Rank {rank}, Iter {iter_num}: Skipped at first-level check (prob={self.style_shift_prob})")
            elif debug_style_shift:
                print(f"[ResNet] Rank {rank} Iter {iter_num} layer1: skip (use_style_shift={self.use_style_shift}, comm={communicator is not None})")

            out2 = self.backbone.layer2(out1)
            if self.use_style_shift and communicator is not None:
                if self.training and random.random() <= self.style_shift_prob:
                    out2 = self.style_shift2(out2, "layer2", communicator, self.training,
                                            verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                    # StyleExplore unconditionally follows StyleShift
                    out2 = self.style_explore2(out2, "layer2", self.training,
                                              verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                    # MixStyle follows StyleExplore
                    out2 = self.mixstyle2(out2, "layer2", self.training,
                                         verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                elif debug_style_shift:
                    print(f"[StyleShift layer2] Rank {rank}, Iter {iter_num}: Skipped at first-level check (prob={self.style_shift_prob})")
            elif debug_style_shift:
                print(f"[ResNet] Rank {rank} Iter {iter_num} layer2: skip (use_style_shift={self.use_style_shift}, comm={communicator is not None})")

            out3 = self.backbone.layer3(out2)
            if self.use_style_shift and communicator is not None:
                if self.training and random.random() <= self.style_shift_prob:
                    out3 = self.style_shift3(out3, "layer3", communicator, self.training,
                                            verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                    # StyleExplore unconditionally follows StyleShift
                    out3 = self.style_explore3(out3, "layer3", self.training,
                                              verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                    # MixStyle follows StyleExplore
                    out3 = self.mixstyle3(out3, "layer3", self.training,
                                         verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                elif debug_style_shift:
                    print(f"[StyleShift layer3] Rank {rank}, Iter {iter_num}: Skipped at first-level check (prob={self.style_shift_prob})")
            elif debug_style_shift:
                print(f"[ResNet] Rank {rank} Iter {iter_num} layer3: skip (use_style_shift={self.use_style_shift}, comm={communicator is not None})")
            
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
            if self.use_style_shift and communicator is not None:
                x = self.style_shift1(x, "layer1", communicator, self.training,
                                     verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # StyleExplore unconditionally follows StyleShift
                x = self.style_explore1(x, "layer1", self.training,
                                      verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # MixStyle follows StyleExplore
                x = self.mixstyle1(x, "layer1", self.training,
                                  verbose=debug_style_shift, iter_num=iter_num, rank=rank)
            
            x = self.backbone.layer2(x)
            if self.use_style_shift and communicator is not None:
                x = self.style_shift2(x, "layer2", communicator, self.training,
                                     verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # StyleExplore unconditionally follows StyleShift
                x = self.style_explore2(x, "layer2", self.training,
                                      verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # MixStyle follows StyleExplore
                x = self.mixstyle2(x, "layer2", self.training,
                                  verbose=debug_style_shift, iter_num=iter_num, rank=rank)
            
            x = self.backbone.layer3(x)
            if self.use_style_shift and communicator is not None:
                x = self.style_shift3(x, "layer3", communicator, self.training,
                                     verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # StyleExplore unconditionally follows StyleShift
                x = self.style_explore3(x, "layer3", self.training,
                                      verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # MixStyle follows StyleExplore
                x = self.mixstyle3(x, "layer3", self.training,
                                  verbose=debug_style_shift, iter_num=iter_num, rank=rank)
            
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            x = x.view(x.size(0), -1)
            logits = self.backbone.fc(x)
            return logits

if __name__ == '__main__':
    net=ResNet(50, 10)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
