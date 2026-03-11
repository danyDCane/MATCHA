import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init  # 補上這行避免 conv_init 報錯
from torch.autograd import Variable
import sys
import numpy as np
from style_transforms import StyleShift, StyleExplore, MixStyle
import random


class CosineClassifier(nn.Module):
    """
    Cosine classifier head:
    - Learnable weight matrix W in R^{num_classes x in_features}
    - Both features and weights are L2-normalized
    - Logits = s * cos(theta), where each row of W acts as a class prototype on the hypersphere.
    """

    def __init__(self, in_features: int, num_classes: int, scale: float = 30.0, learn_scale: bool = False):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))

        if learn_scale:
            self.s = nn.Parameter(torch.tensor(scale, dtype=torch.float))
        else:
            # buffer 確保與模型一起移動到正確裝置與儲存到 state_dict
            self.register_buffer("s", torch.tensor(scale, dtype=torch.float))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_features]
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm)  # [B, num_classes]
        return self.s * cosine

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
    def __init__(
        self,
        depth,
        num_classes,
        use_style_shift=False,
        style_shift_prob=0.5,
        style_shift_ratio=0.5,
        style_explore_alpha=3.0,
        style_explore_ratio=0.5,
        mixstyle_alpha=0.1,
        use_cosine_classifier: bool = False,
        cosine_scale: float = 30.0,
        cosine_learn_scale: bool = False,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.style_shift_prob = style_shift_prob  # 保存为实例属性
        self.style_shift_ratio = style_shift_ratio  # 也可以保存（如果需要）
        self.use_style_shift = use_style_shift
        self.style_explore_alpha = style_explore_alpha
        self.style_explore_ratio = style_explore_ratio
        self.mixstyle_alpha = mixstyle_alpha

        # Cosine classifier 相關設定
        self.use_cosine_classifier = use_cosine_classifier
        self.cosine_scale = cosine_scale
        self.cosine_learn_scale = cosine_learn_scale

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        feat_dim = 64 * block.expansion
        if self.use_cosine_classifier:
            self.linear = CosineClassifier(
                feat_dim,
                num_classes,
                scale=self.cosine_scale,
                learn_scale=self.cosine_learn_scale,
            )
        else:
            self.linear = nn.Linear(feat_dim, num_classes)
        
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
    
    def forward(self, x, communicator=None, debug_style_shift=False, iter_num=-1, rank=-1, return_feature: bool = False, force_style_shift: bool = False):
        """
        Forward pass.

        Args:
            x: input tensor [B, 3, H, W]
            communicator: Communicator object with neighbor_style_stats attribute
                          (used for style shift if enabled)
            debug_style_shift: If True, print debug information for style shift
            iter_num: Current iteration number (for debugging)
            rank: Current rank (for debugging)
            return_feature: if True, also return the final flattened feature
                            vector after style shift (if enabled), with shape
                            [B, C].
            force_style_shift: if True, always apply style shift (skip random check)

        Returns:
            If return_feature is False (default):
                logits: [B, num_classes]
            If return_feature is True:
                logits: [B, num_classes]
                feat: flattened feature vector [B, C].
        """
        out = F.relu(self.bn1(self.conv1(x)))

        # First three convolutional blocks with optional style shift
        out1 = self.layer1(out)
        if self.use_style_shift and communicator is not None:
            if force_style_shift or (self.training and random.random() <= self.style_shift_prob):
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
            if force_style_shift or (self.training and random.random() <= self.style_shift_prob):
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
            if force_style_shift or (self.training and random.random() <= self.style_shift_prob):
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

        if return_feature:
            return logits, feat

        return logits

class StandardResNetWrapper(nn.Module):
    """
    Wrapper for torchvision ResNet with optional style shift and feature
    extraction utilities.
    """

    def __init__(
        self,
        depth,
        num_classes,
        use_style_shift: bool = False,
        style_shift_prob: float = 0.5,
        style_shift_ratio: float = 0.5,
        style_explore_alpha: float = 3.0,
        style_explore_ratio: float = 0.5,
        mixstyle_alpha: float = 0.1,
        pretrained: bool = False,
        use_cosine_classifier: bool = False,
        cosine_scale: float = 30.0,
        cosine_learn_scale: bool = False,
    ):
        super(StandardResNetWrapper, self).__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
        self.style_shift_prob = style_shift_prob  # 保存为实例属性
        self.style_shift_ratio = style_shift_ratio  # 也可以保存（如果需要）
        self.use_style_shift = use_style_shift
        self.style_explore_alpha = style_explore_alpha
        self.style_explore_ratio = style_explore_ratio
        self.mixstyle_alpha = mixstyle_alpha
        self.use_cosine_classifier = use_cosine_classifier
        self.cosine_scale = cosine_scale
        self.cosine_learn_scale = cosine_learn_scale
        
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
        
        # 創建 ResNet 模型（支持预训练权重）
        # 兼容新版本 torchvision (>=0.13) 和旧版本 (<0.13)
        try:
            # Try new API (torchvision >= 0.13)
            if pretrained:
                from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
                weights_dict = {
                    18: ResNet18_Weights.IMAGENET1K_V1,
                    34: ResNet34_Weights.IMAGENET1K_V1,
                    50: ResNet50_Weights.IMAGENET1K_V1,
                    101: ResNet101_Weights.IMAGENET1K_V1,
                    152: ResNet152_Weights.IMAGENET1K_V1
                }
                self.backbone = resnet_dict[depth](weights=weights_dict[depth])
            else:
                self.backbone = resnet_dict[depth](weights=None)
        except (ImportError, AttributeError):
            # Fall back to old API (torchvision < 0.13)
            self.backbone = resnet_dict[depth](pretrained=pretrained)

        # 修改最後一層以匹配 num_classes，並可選擇使用 cosine classifier
        feat_dim = self.backbone.fc.in_features
        if self.use_cosine_classifier:
            self.backbone.fc = CosineClassifier(
                feat_dim,
                num_classes,
                scale=self.cosine_scale,
                learn_scale=self.cosine_learn_scale,
            )
        else:
            self.backbone.fc = nn.Linear(feat_dim, num_classes)
        
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
    
    def intermediate_forward(self, x):
        """
        Extract intermediate features, returning 512-dim feature vector (before fc layer).
        Used for OOD detection with diffusion model.
        
        Args:
            x: input tensor [B, 3, H, W]
        
        Returns:
            features: flattened feature vector (B, 512)
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
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
    
    def forward(self, x, communicator=None, debug_style_shift=False, iter_num=-1, rank=-1, return_feature: bool = False, force_style_shift: bool = False):
        """
        Forward pass with optional style shift and optional final feature output.

        Args:
            x: input tensor [B, 3, H, W]
            communicator: Communicator object with neighbor_style_stats attribute
                          (used for style shift if enabled)
            debug_style_shift: If True, print debug information for style shift
            iter_num: Current iteration number (for debugging)
            rank: Current rank (for debugging)
            return_feature: if True, also return the final flattened feature
                            vector after style shift (if enabled), with shape
                            [B, 512] for ResNet18/34.
            force_style_shift: if True, always apply style shift (skip random check)

        Returns:
            If return_feature is False:
                logits: [B, num_classes]
            If return_feature is True:
                logits: [B, num_classes]
                feat: flattened feature vector [B, 512] (same format as
                      intermediate_forward).
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # Main residual layers with optional style shift
        x = self.backbone.layer1(x)
        if self.use_style_shift and communicator is not None:
            if force_style_shift or (self.training and random.random() <= self.style_shift_prob):
                x = self.style_shift1(x, "layer1", communicator, self.training,
                                      verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # StyleExplore unconditionally follows StyleShift
                x = self.style_explore1(x, "layer1", self.training,
                                        verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # MixStyle follows StyleExplore
                x = self.mixstyle1(x, "layer1", self.training,
                                   verbose=debug_style_shift, iter_num=iter_num, rank=rank)
            elif debug_style_shift:
                print(f"[StyleShift layer1] Rank {rank}, Iter {iter_num}: Skipped at first-level check (prob={self.style_shift_prob})")
        elif debug_style_shift:
            print(f"[ResNet] Rank {rank} Iter {iter_num} layer1: skip (use_style_shift={self.use_style_shift}, comm={communicator is not None})")

        x = self.backbone.layer2(x)
        if self.use_style_shift and communicator is not None:
            if force_style_shift or (self.training and random.random() <= self.style_shift_prob):
                x = self.style_shift2(x, "layer2", communicator, self.training,
                                      verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # StyleExplore unconditionally follows StyleShift
                x = self.style_explore2(x, "layer2", self.training,
                                        verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # MixStyle follows StyleExplore
                x = self.mixstyle2(x, "layer2", self.training,
                                   verbose=debug_style_shift, iter_num=iter_num, rank=rank)
            elif debug_style_shift:
                print(f"[StyleShift layer2] Rank {rank}, Iter {iter_num}: Skipped at first-level check (prob={self.style_shift_prob})")
        elif debug_style_shift:
            print(f"[ResNet] Rank {rank} Iter {iter_num} layer2: skip (use_style_shift={self.use_style_shift}, comm={communicator is not None})")

        x = self.backbone.layer3(x)
        if self.use_style_shift and communicator is not None:
            if force_style_shift or (self.training and random.random() <= self.style_shift_prob):
                x = self.style_shift3(x, "layer3", communicator, self.training,
                                      verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # StyleExplore unconditionally follows StyleShift
                x = self.style_explore3(x, "layer3", self.training,
                                        verbose=debug_style_shift, iter_num=iter_num, rank=rank)
                # MixStyle follows StyleExplore
                x = self.mixstyle3(x, "layer3", self.training,
                                   verbose=debug_style_shift, iter_num=iter_num, rank=rank)
            elif debug_style_shift:
                print(f"[StyleShift layer3] Rank {rank}, Iter {iter_num}: Skipped at first-level check (prob={self.style_shift_prob})")
        elif debug_style_shift:
            print(f"[ResNet] Rank {rank} Iter {iter_num} layer3: skip (use_style_shift={self.use_style_shift}, comm={communicator is not None})")
        
        x = self.backbone.layer4(x)
        
        # Global average pooling and classifier head
        feat = self.backbone.avgpool(x)
        feat = feat.view(feat.size(0), -1)
        logits = self.backbone.fc(feat)

        if return_feature:
            return logits, feat

        return logits

if __name__ == '__main__':
    net=ResNet(50, 10)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
