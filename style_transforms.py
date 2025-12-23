import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, Tuple, Optional
from torch.distributions import Beta


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


class StyleExplore(nn.Module):
    """
    Style Explore module that performs style extrapolation to generate novel, extreme styles.
    
    This module:
    1. Calculates instance-level and batch-level average style statistics
    2. Selects a subset of samples to modify
    3. Extrapolates the selected samples' styles away from the batch center
    4. Applies AdaIN to transfer the extrapolated styles
    5. Replaces the original samples with the modified ones
    """
    
    def __init__(self, alpha: float = 3.0, explore_ratio: float = 0.5):
        """
        Args:
            alpha: Extrapolation coefficient. The paper recommends alpha = 3.0
            explore_ratio: Fixed ratio of samples in the batch to be modified (0.0 to 1.0)
        """
        super(StyleExplore, self).__init__()
        self.alpha = alpha
        self.explore_ratio = explore_ratio
    
    def forward(
        self,
        x: torch.Tensor,
        layer_name: str = "",
        training: bool = True,
        verbose: bool = False,
        iter_num: int = -1,
        rank: int = -1
    ) -> torch.Tensor:
        """
        Apply style exploration (extrapolation) to features.
        
        Args:
            x: Input feature map of shape [B, C, H, W]
            layer_name: Name of the layer (e.g., "layer1", "layer2", "layer3")
            training: Whether in training mode
            verbose: If True, print debug information
            iter_num: Current iteration number (for debugging)
            rank: Current rank (for debugging)
        
        Returns:
            Features with style exploration applied (or original features if not applied)
        """
        # Step 1: Only apply during training
        if not training:
            if verbose:
                print(f"[StyleExplore {layer_name}] Skipped: not in training mode")
            return x
        
        B, C, H, W = x.shape
        eps = 1e-5
        device = x.device
        
        # Step 2: Statistics Calculation
        # Calculate instance-level statistics for every sample in the batch
        x_flat = x.view(B, C, -1)  # [B, C, H*W]
        
        # μ_original: instance-level mean for each sample [B, C]
        mu_original = x_flat.mean(dim=2)  # [B, C]
        
        # σ_original: instance-level std for each sample [B, C]
        var_original = x_flat.var(dim=2, unbiased=False)  # [B, C]
        sigma_original = torch.sqrt(var_original + eps)  # [B, C]
        
        # Calculate batch-level average style statistics
        # μ_batch_avg = mean(μ_original, dim=0) [C]
        mu_batch_avg = mu_original.mean(dim=0)  # [C]
        
        # σ_batch_avg = mean(σ_original, dim=0) [C]
        sigma_batch_avg = sigma_original.mean(dim=0)  # [C]
        
        # Detach batch averages from computation graph (used as reference point)
        mu_batch_avg_detached = mu_batch_avg.detach()
        sigma_batch_avg_detached = sigma_batch_avg.detach()
        
        # Step 3: Selection
        # Determine number of samples to explore: B_e
        B_e = max(1, int(B * self.explore_ratio))
        
        # Randomly select indices
        explore_indices = torch.randperm(B, device=device)[:B_e]
        
        if verbose:
            print(f"[StyleExplore {layer_name}] explore_indices (first 10): {explore_indices[:10].tolist()}")
        
        if len(explore_indices) == 0:
            if verbose:
                print(f"[StyleExplore {layer_name}] Skipped: no samples selected")
            return x
        
        # Step 4: Extrapolation (The "Explore" Step)
        # For selected indices, calculate new target style statistics using extrapolation formula:
        # μ_new = μ_original + α × (μ_original - μ_batch_avg)
        # σ_new = σ_original + α × (σ_original - σ_batch_avg)
        
        # Extract statistics for selected samples
        mu_selected = mu_original[explore_indices]  # [B_e, C]
        sigma_selected = sigma_original[explore_indices]  # [B_e, C]
        
        # Calculate extrapolated statistics
        # μ_new = μ_original + alpha * (μ_original - μ_batch_avg)
        mu_new = mu_selected + self.alpha * (mu_selected - mu_batch_avg_detached.unsqueeze(0))  # [B_e, C]
        
        # σ_new = σ_original + alpha * (σ_original - σ_batch_avg)
        sigma_new = sigma_selected + self.alpha * (sigma_selected - sigma_batch_avg_detached.unsqueeze(0))  # [B_e, C]
        
        # Ensure sigma_new is positive
        sigma_new = torch.clamp(sigma_new, min=eps)
        
        if verbose:
            d_mu_before = (mu_selected - mu_batch_avg_detached.unsqueeze(0)).norm(dim=1).mean()
            d_mu_after  = (mu_new      - mu_batch_avg_detached.unsqueeze(0)).norm(dim=1).mean()
            print(f"[StyleExplore {layer_name}] extrapolation_ratio(mu): "
                f"{(d_mu_after/(d_mu_before+1e-12)).item():.3f} (expected {1+self.alpha:.3f})")

            d_s_before = (sigma_selected - sigma_batch_avg_detached.unsqueeze(0)).norm(dim=1).mean()
            d_s_after  = (sigma_new      - sigma_batch_avg_detached.unsqueeze(0)).norm(dim=1).mean()
            print(f"[StyleExplore {layer_name}] extrapolation_ratio(std): "
                f"{(d_s_after/(d_s_before+1e-12)).item():.3f} (expected {1+self.alpha:.3f})")
        
        # Step 5: Style Transfer
        # Extract features to modify
        x_selected = x[explore_indices]  # [B_e, C, H, W]
        
        # Apply AdaIN in batch (more efficient than looping)
        # Since each sample has unique mu_new and sigma_new, we process them together
        # using broadcasting: mu_new [B_e, C] and sigma_new [B_e, C]
        x_selected_flat = x_selected.view(B_e, C, -1)  # [B_e, C, H*W]
        
        # Compute content statistics (per sample, per channel)
        content_mean = x_selected_flat.mean(dim=2)  # [B_e, C]
        content_var = x_selected_flat.var(dim=2, unbiased=False)  # [B_e, C]
        content_std = torch.sqrt(content_var + eps)  # [B_e, C]
        
        # Normalize content: (x - mean) / std
        content_norm = (x_selected_flat - content_mean.unsqueeze(2)) / content_std.unsqueeze(2)  # [B_e, C, H*W]
        
        # Apply target style: normalized * target_std + target_mean
        # mu_new and sigma_new are [B_e, C], expand to [B_e, C, 1] for broadcasting
        mu_new_expanded = mu_new.unsqueeze(2)  # [B_e, C, 1]
        sigma_new_expanded = sigma_new.unsqueeze(2)  # [B_e, C, 1]
        
        x_explored_flat = content_norm * sigma_new_expanded + mu_new_expanded  # [B_e, C, H*W]
        x_explored = x_explored_flat.view(B_e, C, H, W)  # [B_e, C, H, W]
        
        if verbose:
            # Compute stats before/after for debugging
            def _mean_std_per_sample(z):
                Be, C, H, W = z.shape
                zf = z.view(Be, C, -1)
                mu = zf.mean(dim=2)  # [Be, C]
                var = zf.var(dim=2, unbiased=False)
                std = torch.sqrt(var + eps)  # [Be, C]
                return mu, std

            mu_a, std_a = _mean_std_per_sample(x_explored.detach())      # [Be, C]
            # 目標就是 mu_new / sigma_new（也是 [Be, C]）
            rel_mu = (mu_a - mu_new).norm() / (mu_new.norm() + 1e-12)
            rel_std = (std_a - sigma_new).norm() / (sigma_new.norm() + 1e-12)
            
            delta = (x_explored.detach() - x_selected.detach()).abs().mean()
            has_nan = torch.isnan(x_explored).any().item()
            has_inf = torch.isinf(x_explored).any().item()
            
            print(f"[StyleExplore {layer_name}] AdaIN check: "
                  f"rel_mu={rel_mu.item():.3e}, rel_std={rel_std.item():.3e}, "
                  f"mean_abs_delta={delta.item():.3e}, nan={has_nan}, inf={has_inf}")
        
        # Step 6: Merge results
        # Create output copy and replace selected samples
        output = x.clone()
        output[explore_indices] = x_explored
        
        if verbose:
            print(f"[StyleExplore {layer_name}] Rank {rank}, Iter {iter_num}: "
                  f"Applied to {len(explore_indices)}/{B} samples, "
                  f"alpha={self.alpha}, "
                  f"mu_new_range=[{mu_new.min():.4f}, {mu_new.max():.4f}], "
                  f"sigma_new_range=[{sigma_new.min():.4f}, {sigma_new.max():.4f}]")
            print(f"[StyleExplore {layer_name}] mu_new_norm_mean={mu_new.norm(dim=1).mean().item():.3f} "
                  f"sigma_new_norm_mean={sigma_new.norm(dim=1).mean().item():.3f}")

        return output


class MixStyle(nn.Module):
    """
    MixStyle module that mixes style statistics within a batch.
    
    This module:
    1. Calculates per-sample, per-channel statistics μ(x) and σ(x) for the current batch
    2. Generates a random permutation to get paired samples' statistics
    3. Samples a mixing coefficient λ from Beta(α, α) distribution (batch-wise, shared)
    4. Computes mixed statistics: μ_mix = λ * μ(x) + (1-λ) * μ(x_perm)
    5. Applies AdaIN to transfer the mixed style
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Parameter for Beta distribution Beta(α, α). Default is 0.1 as recommended.
        """
        super(MixStyle, self).__init__()
        self.alpha = alpha
    
    def forward(
        self,
        x: torch.Tensor,
        layer_name: str = "",
        training: bool = True,
        verbose: bool = False,
        iter_num: int = -1,
        rank: int = -1
    ) -> torch.Tensor:
        """
        Apply MixStyle to features.
        
        Args:
            x: Input feature map of shape [B, C, H, W]
            layer_name: Name of the layer (e.g., "layer1", "layer2", "layer3")
            training: Whether in training mode
            verbose: If True, print debug information
            iter_num: Current iteration number (for debugging)
            rank: Current rank (for debugging)
        
        Returns:
            Features with MixStyle applied (or original features if not applied)
        """
        # Step 1: Only apply during training
        if not training:
            if verbose:
                print(f"[MixStyle {layer_name}] Skipped: not in training mode")
            return x
        
        B, C, H, W = x.shape
        eps = 1e-5
        device = x.device
        
        # Step A: Calculate statistics for current batch
        # Compute per-sample, per-channel statistics μ(x) and σ(x)
        x_flat = x.view(B, C, -1)  # [B, C, H*W]
        
        # μ(x): instance-level mean for each sample [B, C]
        mu = x_flat.mean(dim=2)  # [B, C]
        
        # σ(x): instance-level std for each sample [B, C]
        var = x_flat.var(dim=2, unbiased=False)  # [B, C]
        sigma = torch.sqrt(var + eps)  # [B, C]
        
        # Step B: Permutation - generate random permutation indices
        perm_indices = torch.randperm(B, device=device)
        
        # Get paired samples' statistics μ(x_perm) and σ(x_perm)
        mu_perm = mu[perm_indices]  # [B, C]
        sigma_perm = sigma[perm_indices]  # [B, C]
        
        # Step C: Sample mixing coefficient λ from Beta(α, α)
        # Batch-wise: the entire batch shares the same λ
        dist = Beta(self.alpha, self.alpha)
        lam = dist.sample().item()  # Single scalar for the entire batch
        
        # Step D: Compute mixed statistics and apply AdaIN
        # μ_mix = λ * μ(x) + (1-λ) * μ(x_perm)
        mu_mix = lam * mu + (1 - lam) * mu_perm  # [B, C]
        
        # σ_mix = λ * σ(x) + (1-λ) * σ(x_perm)
        sigma_mix = lam * sigma + (1 - lam) * sigma_perm  # [B, C]
        
        # Ensure sigma_mix is positive
        sigma_mix = torch.clamp(sigma_mix, min=eps)
        
        if verbose:
            print(f"[MixStyle {layer_name}] Rank {rank}, Iter {iter_num}: "
                  f"lambda={lam:.4f}, "
                  f"mu_range=[{mu_mix.min():.4f}, {mu_mix.max():.4f}], "
                  f"sigma_range=[{sigma_mix.min():.4f}, {sigma_mix.max():.4f}]")
        
        # Apply AdaIN in batch (efficient processing)
        # Since each sample has unique mu_mix and sigma_mix, we process them together
        # using broadcasting: mu_mix [B, C] and sigma_mix [B, C]
        
        # Compute content statistics (per sample, per channel)
        content_mean = mu  # [B, C] - already computed
        content_std = sigma  # [B, C] - already computed
        
        # Normalize content: (x - mean) / std
        content_norm = (x_flat - content_mean.unsqueeze(2)) / content_std.unsqueeze(2)  # [B, C, H*W]
        
        # Apply target style: normalized * target_std + target_mean
        # mu_mix and sigma_mix are [B, C], expand to [B, C, 1] for broadcasting
        mu_mix_expanded = mu_mix.unsqueeze(2)  # [B, C, 1]
        sigma_mix_expanded = sigma_mix.unsqueeze(2)  # [B, C, 1]
        
        x_mixed_flat = content_norm * sigma_mix_expanded + mu_mix_expanded  # [B, C, H*W]
        x_mixed = x_mixed_flat.view(B, C, H, W)  # [B, C, H, W]
        
        if verbose:
            # Compute stats before/after for debugging
            def _mean_std_per_sample(z):
                Bz, Cz, Hz, Wz = z.shape
                zf = z.view(Bz, Cz, -1)
                mu_z = zf.mean(dim=2)  # [Bz, Cz]
                var_z = zf.var(dim=2, unbiased=False)
                std_z = torch.sqrt(var_z + eps)  # [Bz, Cz]
                return mu_z, std_z

            mu_a, std_a = _mean_std_per_sample(x_mixed.detach())  # [B, C]
            # Target is mu_mix / sigma_mix (also [B, C])
            rel_mu = (mu_a - mu_mix).norm() / (mu_mix.norm() + 1e-12)
            rel_std = (std_a - sigma_mix).norm() / (sigma_mix.norm() + 1e-12)
            
            delta = (x_mixed.detach() - x.detach()).abs().mean()
            has_nan = torch.isnan(x_mixed).any().item()
            has_inf = torch.isinf(x_mixed).any().item()
            
            print(f"[MixStyle {layer_name}] AdaIN check: "
                  f"rel_mu={rel_mu.item():.3e}, rel_std={rel_std.item():.3e}, "
                  f"mean_abs_delta={delta.item():.3e}, nan={has_nan}, inf={has_inf}")

        return x_mixed

