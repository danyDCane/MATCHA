import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


def compute_layer_style_stats(
    feat: torch.Tensor,
    eta: float = 1e-5,
) -> Dict[str, torch.Tensor]:
    """
    Compute STYLEDDG-style statistics for a single convolutional feature map.

    Args:
        feat: Tensor of shape [B, C, H, W]
        eta: Small constant for numerical stability.

    Returns:
        A dict with keys:
            - mu_bar:           [C]  (batch + spatial mean for each channel)
            - sigma_bar:        [C]  (batch + spatial std for each channel)
            - Sigma_mu_sq:      [C]  (variance of per-sample channel means over batch)
            - Sigma_sigma_sq:   [C]  (variance of per-sample channel stds over batch)

    The semantics are aligned with the notation in STYLEDDG (file://STYLEDDG.pdf),
    where for client i and layer ℓ, the four style statistics are
    \bar{μ}_{θ,ℓ}(x_i), \bar{σ}_{θ,ℓ}(x_i), Σ_{μ,θ,ℓ}^2(x_i), Σ_{σ,θ,ℓ}^2(x_i).
    """
    if feat.dim() != 4:
        raise ValueError(f"Expected feat with shape [B, C, H, W], got {feat.shape}")

    B, C, H, W = feat.shape
    # [B, C, S], S = H * W
    feat_flat = feat.view(B, C, -1)

    # Per-sample, per-channel spatial mean: [B, C]
    mu_sample = feat_flat.mean(dim=2)

    # Per-sample, per-channel spatial std: [B, C]
    var_sample = feat_flat.var(dim=2, unbiased=False)
    sigma_sample = torch.sqrt(var_sample + eta)

    # Batch-level averages: [C]
    mu_bar = mu_sample.mean(dim=0)
    sigma_bar = sigma_sample.mean(dim=0)

    # Batch-level variances of per-sample stats: [C]
    Sigma_mu_sq = mu_sample.var(dim=0, unbiased=False) + eta
    Sigma_sigma_sq = sigma_sample.var(dim=0, unbiased=False) + eta

    return {
        "mu_bar": mu_bar,                 # [C]
        "sigma_bar": sigma_bar,           # [C]
        "Sigma_mu_sq": Sigma_mu_sq,       # [C]
        "Sigma_sigma_sq": Sigma_sigma_sq  # [C]
    }


def compute_multi_layer_style_stats(
    features: Dict[str, torch.Tensor],
    eta: float = 1e-5,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute style statistics for multiple layers.

    Args:
        features: dict mapping layer_name -> feature map [B, C, H, W]
        eta: numerical stability constant.

    Returns:
        Dict[layer_name] -> dict of 4 statistics from `compute_layer_style_stats`.
    """
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, feat in features.items():
        stats[name] = compute_layer_style_stats(feat, eta=eta)
    return stats


def flatten_style_stats(
    stats: Dict[str, Dict[str, torch.Tensor]],
    layer_order: List[str],
) -> torch.Tensor:
    """
    Flatten multi-layer style statistics into a single 1D style vector,
    suitable for communication / style sharing.

    Concatenation order for each layer:
        [mu_bar, sigma_bar, Sigma_mu_sq, Sigma_sigma_sq]

    Args:
        stats: output of `compute_multi_layer_style_stats`.
        layer_order: the order of layers to concatenate, e.g. ["layer1", "layer2", "layer3"].

    Returns:
        style_vec: 1D tensor containing concatenated stats from all specified layers.
    """
    vecs: List[torch.Tensor] = []

    for layer_name in layer_order:
        if layer_name not in stats:
            raise KeyError(f"Layer '{layer_name}' not found in stats.")
        s = stats[layer_name]
        vecs.extend(
            [
                s["mu_bar"].reshape(-1),
                s["sigma_bar"].reshape(-1),
                s["Sigma_mu_sq"].reshape(-1),
                s["Sigma_sigma_sq"].reshape(-1),
            ]
        )

    return torch.cat(vecs, dim=0)


def unflatten_style_stats(
    style_vec: torch.Tensor,
    layer_order: List[str],
    channels_per_layer: Dict[str, int],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Unflatten a 1D style vector back into the original multi-layer style statistics format.
    
    This is the inverse operation of flatten_style_stats.
    
    Args:
        style_vec: 1D tensor containing concatenated stats from all layers
        layer_order: the order of layers, e.g. ["layer1", "layer2", "layer3"]
        channels_per_layer: dict mapping layer_name -> number of channels, e.g. {"layer1": 64, "layer2": 128, "layer3": 256}
    
    Returns:
        Dict mapping layer_name -> dict of 4 statistics:
            {
                "layer1": {"mu_bar": [C1], "sigma_bar": [C1], "Sigma_mu_sq": [C1], "Sigma_sigma_sq": [C1]},
                "layer2": {"mu_bar": [C2], "sigma_bar": [C2], "Sigma_mu_sq": [C2], "Sigma_sigma_sq": [C2]},
                ...
            }
    """
    stats = {}
    offset = 0
    
    for layer_name in layer_order:
        if layer_name not in channels_per_layer:
            raise KeyError(f"Channel count for layer '{layer_name}' not found in channels_per_layer")
        
        C = channels_per_layer[layer_name]
        # Each layer has 4 statistics, each of size C
        layer_size = 4 * C
        
        # Extract this layer's portion from style_vec
        layer_vec = style_vec[offset:offset + layer_size]
        
        # Split into 4 statistics
        mu_bar = layer_vec[0*C:(0+1)*C]
        sigma_bar = layer_vec[1*C:(1+1)*C]
        Sigma_mu_sq = layer_vec[2*C:(2+1)*C]
        Sigma_sigma_sq = layer_vec[3*C:(3+1)*C]
        
        stats[layer_name] = {
            "mu_bar": mu_bar,
            "sigma_bar": sigma_bar,
            "Sigma_mu_sq": Sigma_mu_sq,
            "Sigma_sigma_sq": Sigma_sigma_sq,
        }
        
        offset += layer_size
    
    return stats


def extract_resnet_block_features(
    model: nn.Module,
    x: torch.Tensor,
    blocks: Tuple[str, ...] = ("layer1", "layer2", "layer3"),
) -> Dict[str, torch.Tensor]:
    """
    Extract outputs of specified ResNet blocks without modifying the model definition.

    This is tailored for the MATCHA `ResNet` in `models/resnet.py`, which
    has attributes: conv1, bn1, layer1, layer2, layer3.

    Args:
        model: ResNet-like model with attributes conv1, bn1, and given blocks.
        x: input tensor [B, 3, H, W].
        blocks: tuple of block names to record, in forward order.

    Returns:
        features: dict mapping block_name -> feature map [B, C, H, W].
    """
    # Basic sanity checks (soft, to keep it pluggable)
    if not hasattr(model, "conv1") or not hasattr(model, "bn1"):
        raise AttributeError("Model must have attributes 'conv1' and 'bn1'.")

    # Initial stem: conv1 + bn1 + ReLU
    out = F.relu(model.bn1(model.conv1(x)))

    features: Dict[str, torch.Tensor] = {}

    # Sequentially pass through blocks, recording requested ones
    for block_name in ["layer1", "layer2", "layer3"]:
        if not hasattr(model, block_name):
            continue
        block = getattr(model, block_name)
        out = block(out)
        if block_name in blocks:
            features[block_name] = out

    return features


# def get_resnet_style_vector(
#     model: nn.Module,
#     x: torch.Tensor,
#     eta: float = 1e-5,
#     blocks: Tuple[str, ...] = ("layer1", "layer2", "layer3"),
# ) -> torch.Tensor:
#     """
#     High-level helper: for a MATCHA ResNet and a batch x, directly produce
#     a flattened style vector for the specified blocks.

#     Steps:
#         1) Extract block features
#         2) Compute per-layer style stats
#         3) Flatten into a single 1D vector

#     Args:
#         model: ResNet-like model.
#         x: input tensor [B, 3, H, W].
#         eta: numerical stability constant.
#         blocks: which blocks to consider, default first three conv blocks.

#     Returns:
#         style_vec: 1D tensor suitable for style sharing.
#     """
#     with torch.no_grad():
#         feats = extract_resnet_block_features(model, x, blocks=blocks)
#         stats = compute_multi_layer_style_stats(feats, eta=eta)
#         style_vec = flatten_style_stats(stats, layer_order=list(blocks))
#     return style_vec


__all__ = [
    "compute_layer_style_stats",
    "compute_multi_layer_style_stats",
    "flatten_style_stats",
    "unflatten_style_stats",
    "extract_resnet_block_features",
    # "get_resnet_style_vector",
]


