import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, Tuple, Optional


def adain(content: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Adaptive Instance Normalization (AdaIN).
    
    Transfers the style of target_mean and target_std to the content features.
    The content features are normalized by their own statistics, then rescaled
    and shifted by the target statistics.
    
    Args:
        content: Content features of shape [B, C, H, W]
        target_mean: Target mean statistics of shape [C]
        target_std: Target std statistics of shape [C]
        eps: Small constant for numerical stability
    
    Returns:
        Styled features of shape [B, C, H, W]
    """
    B, C, H, W = content.shape
    
    # Compute content statistics (per sample, per channel)
    content_flat = content.view(B, C, -1)  # [B, C, H*W]
    content_mean = content_flat.mean(dim=2)  # [B, C]
    content_var = content_flat.var(dim=2, unbiased=False)          # [B, C]
    content_std = torch.sqrt(content_var + eps)                   # [B, C]

    
    # Normalize content: (x - mean) / std
    content_norm = (content_flat - content_mean.unsqueeze(2)) / content_std.unsqueeze(2)
    
    # Apply target style: normalized * target_std + target_mean
    target_mean_expanded = target_mean.unsqueeze(0).unsqueeze(2)  # [1, C, 1]
    target_std_expanded = target_std.unsqueeze(0).unsqueeze(2)  # [1, C, 1]
    styled = content_norm * target_std_expanded + target_mean_expanded
    
    return styled.view(B, C, H, W)


def generate_target_style_from_neighbor(
    neighbor_stats: Dict[str, Dict[str, torch.Tensor]],
    layer_name: str,
    device: torch.device,
    verbose: bool = False,
    iter_num: int = -1,
    rank: int = -1,
    eta: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate target style statistics from neighbor's style statistics using DSU mechanism.
    
    Uses the neighbor's mean (mu_bar) and variance (Sigma_mu_sq) to sample a random
    target mean, and similarly for the standard deviation.
    
    Formula:
        beta = mu_bar + epsilon * sqrt(Sigma_mu_sq), where epsilon ~ N(0, 1)
        gamma = sigma_bar + epsilon * sqrt(Sigma_sigma_sq)
    
    Args:
        neighbor_stats: Dictionary mapping layer_name to style statistics
            {layer_name: {mu_bar: [C], sigma_bar: [C], Sigma_mu_sq: [C], Sigma_sigma_sq: [C]}}
        layer_name: Name of the layer (e.g., "layer1", "layer2", "layer3")
        device: Device to place the generated tensors on
        eta: Small constant for numerical stability
    
    Returns:
        Tuple of (target_mean, target_std), both of shape [C]
    """
    if layer_name not in neighbor_stats:
        raise KeyError(f"Layer '{layer_name}' not found in neighbor_stats")
    
    layer_stats = neighbor_stats[layer_name]
    mu_bar = layer_stats["mu_bar"].to(device)  # [C]
    sigma_bar = layer_stats["sigma_bar"].to(device)  # [C]
    Sigma_mu_sq = layer_stats["Sigma_mu_sq"].to(device)  # [C]
    Sigma_sigma_sq = layer_stats["Sigma_sigma_sq"].to(device)  # [C]
    
    if verbose:
        print(f"[generate_target_style_from_neighbor] Rank {rank}, Iter {iter_num}: "
              f"mu_bar_range=[{mu_bar.min():.4f}, {mu_bar.max():.4f}], "
              f"sigma_bar_range=[{sigma_bar.min():.4f}, {sigma_bar.max():.4f}], "
              f"Sigma_mu_sq_range=[{Sigma_mu_sq.min():.4f}, {Sigma_mu_sq.max():.4f}], "
              f"Sigma_sigma_sq_range=[{Sigma_sigma_sq.min():.4f}, {Sigma_sigma_sq.max():.4f}]")
    
    # Sample epsilon from standard normal distribution
    epsilon_mu = torch.randn_like(mu_bar)  # [C]
    epsilon_sigma = torch.randn_like(sigma_bar)  # [C]
    
    # Generate target mean: beta = mu_bar + epsilon * sqrt(Sigma_mu_sq)
    target_mean = mu_bar + epsilon_mu * torch.sqrt(Sigma_mu_sq + eta)
    
    # Generate target std: gamma = sigma_bar + epsilon * sqrt(Sigma_sigma_sq)
    target_std = sigma_bar + epsilon_sigma * torch.sqrt(Sigma_sigma_sq + eta)
    
    # Ensure target_std is positive
    target_std = torch.clamp(target_std, min=eta)
    
    return target_mean, target_std


class StyleShift(nn.Module):
    """
    Style Shift module that transforms a portion of the batch to neighbor's style.
    
    This module:
    1. Splits the batch into two parts: one kept original, one to be transformed
    2. Generates target style from neighbor's statistics using DSU mechanism
    3. Applies AdaIN to transform the selected portion
    4. Merges the transformed and original portions
    """
    
    def __init__(self, activation_prob: float = 0.5, shift_ratio: float = 0.5):
        """
        Args:
            activation_prob: Probability of activating the style shift module
            shift_ratio: Ratio of samples in the batch to be transformed (0.0 to 1.0)
        """
        super(StyleShift, self).__init__()
        self.activation_prob = activation_prob
        self.shift_ratio = shift_ratio
    
    def forward(
        self,
        features: torch.Tensor,
        layer_name: str,
        communicator: Optional[object],
        training: bool = True,
        verbose: bool = False,
        iter_num: int = -1,
        rank: int = -1
    ) -> torch.Tensor:
        """
        Apply style shift to features if conditions are met.
        
        Args:
            features: Input features of shape [B, C, H, W]
            layer_name: Name of the layer (e.g., "layer1", "layer2", "layer3")
            communicator: Communicator object with neighbor_style_stats attribute
            training: Whether in training mode
            verbose: If True, print debug information
            iter_num: Current iteration number (for debugging)
            rank: Current rank (for debugging)
        
        Returns:
            Features with style shift applied (or original features if not applied)
        """
        # Only apply during training
        if not training:
            if verbose:
                print(f"[StyleShift {layer_name}] Skipped: not in training mode")
            return features
        
        # Check if communicator and neighbor stats are available
        if communicator is None:
            if verbose:
                print(f"[StyleShift {layer_name}] Skipped: communicator is None")
            return features
        
        if not hasattr(communicator, 'neighbor_style_stats'):
            if verbose:
                print(f"[StyleShift {layer_name}] Skipped: communicator has no neighbor_style_stats")
            return features
        
        if not communicator.neighbor_style_stats:
            if verbose:
                print(f"[StyleShift {layer_name}] Skipped: neighbor_style_stats is empty")
            return features
        
        # Randomly select a neighbor
        neighbor_ranks = list(communicator.neighbor_style_stats.keys())
        if not neighbor_ranks:
            if verbose:
                print(f"[StyleShift {layer_name}] Skipped: no neighbor ranks available")
            return features
        
        selected_neighbor = random.choice(neighbor_ranks)
        neighbor_stats = communicator.neighbor_style_stats[selected_neighbor]
        
        # Check if the layer exists in neighbor stats
        if layer_name not in neighbor_stats:
            if verbose:
                print(f"[StyleShift {layer_name}] Skipped: layer not found in neighbor stats")
            return features
        
        # Generate target style from neighbor using DSU mechanism
        device = features.device
        try:
            target_mean, target_std = generate_target_style_from_neighbor(
                neighbor_stats, layer_name, device
            )
        except Exception as e:
            # If generation fails, return original features
            if verbose:
                print(f"[StyleShift {layer_name}] Error generating target style: {e}")
            return features
        
        # Split batch: determine how many samples to transform
        B = features.size(0)
        num_shift = max(1, int(B * self.shift_ratio))
        
        # Randomly select indices to transform
        shift_indices = torch.randperm(B, device=device)[:num_shift]
        if verbose:
            print(f"[StyleShift {layer_name}] shift_indices (first 10): {shift_indices[:10].tolist()}")


        
        # Apply style shift to selected samples
        if len(shift_indices) > 0:
            features_to_shift = features[shift_indices]  # [num_shift, C, H, W]
            shifted_features = adain(features_to_shift, target_mean, target_std)
            
            if verbose:
                # compute stats before/after on shifted subset
                def _mean_std(z):
                    B2, C2, H2, W2 = z.shape
                    zf = z.view(B2, C2, -1)
                    mu = zf.mean(dim=2).mean(dim=0)  # [C] (avg over batch subset)
                    var = zf.var(dim=2, unbiased=False).mean(dim=0)  # [C]
                    std = torch.sqrt(var + 1e-5)
                    return mu, std

                mu_b, std_b = _mean_std(features_to_shift.detach())
                mu_a, std_a = _mean_std(shifted_features.detach())

                # relative errors
                rel_mu = (mu_a - target_mean).norm() / (target_mean.norm() + 1e-12)
                rel_std = (std_a - target_std).norm() / (target_std.norm() + 1e-12)

                delta = (shifted_features.detach() - features_to_shift.detach()).abs().mean()
                has_nan = torch.isnan(shifted_features).any().item()
                has_inf = torch.isinf(shifted_features).any().item()

                print(f"[StyleShift {layer_name}] AdaIN check: "
                      f"rel_mu={rel_mu.item():.3e}, rel_std={rel_std.item():.3e}, "
                      f"mean_abs_delta={delta.item():.3e}, nan={has_nan}, inf={has_inf}")

            # Merge: create output tensor with original shape
            output = features.clone()
            output[shift_indices] = shifted_features
            
            if verbose:
                print(f"[StyleShift {layer_name}] Rank {rank}, Iter {iter_num}: "
                      f"Applied to {len(shift_indices)}/{B} samples, "
                      f"neighbor={selected_neighbor}, "
                      f"target_mean_range=[{target_mean.min():.4f}, {target_mean.max():.4f}], "
                      f"target_std_range=[{target_std.min():.4f}, {target_std.max():.4f}]")
            
            return output
        
        return features

