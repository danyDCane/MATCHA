"""V2-B-1 score-norm regularizer gradient-conflict diagnostics.

只在 --enable_score_reg_diag 開啟時跑。對同一個 forward graph 分別計算 backbone 端的
loss_cls 與 lambda_eff * loss_reg 梯度，量兩者範數與 cosine——若 cos < -0.3 持續，
表示 regularizer 與分類路徑方向衝突，會解釋 R1 卡在邊際或退步的根因。

使用 torch.autograd.grad（retain_graph=True）取代 .backward()，
不污染 .grad、不消耗主訓練的 backward 路徑。
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


def _flatten_grads(grads, params: List[nn.Parameter]) -> torch.Tensor:
    flats = []
    for g, p in zip(grads, params):
        if g is None:
            flats.append(p.new_zeros(p.numel()))
        else:
            flats.append(g.detach().flatten())
    return torch.cat(flats)


def compute_grad_conflict(
    loss_cls: torch.Tensor,
    loss_reg_weighted: torch.Tensor,
    params: List[nn.Parameter],
) -> Dict[str, float]:
    """Compute backbone gradient conflict between cls and weighted reg losses.

    Args:
        loss_cls: scalar tensor (CE loss with backbone graph).
        loss_reg_weighted: scalar tensor (lambda_eff * loss_reg with backbone graph).
        params: backbone params to compute gradients w.r.t. (must require grad).

    Returns:
        dict with g_cls_norm, g_reg_norm, g_cos.
    """
    # Must retain graph because the main loss.backward() runs after this.
    grad_cls = torch.autograd.grad(
        loss_cls, params, retain_graph=True, create_graph=False, allow_unused=True,
    )
    grad_reg = torch.autograd.grad(
        loss_reg_weighted, params, retain_graph=True, create_graph=False, allow_unused=True,
    )

    g_cls_flat = _flatten_grads(grad_cls, params)
    g_reg_flat = _flatten_grads(grad_reg, params)

    g_cls_norm = float(g_cls_flat.norm().item())
    g_reg_norm = float(g_reg_flat.norm().item())
    denom = g_cls_norm * g_reg_norm
    if denom > 0:
        g_cos = float((g_cls_flat @ g_reg_flat).item() / (denom + 1e-12))
    else:
        g_cos = 0.0

    return {
        "g_cls_norm": g_cls_norm,
        "g_reg_norm": g_reg_norm,
        "g_cos": g_cos,
    }


def compute_feature_collapse_stats(
    feats: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Dimensional-collapse diagnostics for a feature matrix.

    Args:
        feats: [N, D] penultimate features (vec_style). Any device/dtype.
        labels: [N] int class labels (optional). Needed for inter/intra ratio.

    Returns:
        dict with:
          erank       — effective rank = exp(entropy of normalized singular values)
                        of the centered feature matrix. Range (1, D]. Collapse -> down.
          tvar        — total variance = trace(cov) = mean_i ||x_i - mean||^2. Collapse -> down.
          inter_intra — Fisher-like ratio: variance of class centroids / mean within-class
                        variance. None if labels not given.
          n_samples   — N.
    """
    f = feats.detach().to(torch.float64).cpu()
    n = f.size(0)
    mean = f.mean(dim=0, keepdim=True)
    centered = f - mean

    tvar = float(centered.pow(2).sum(dim=1).mean().item())

    # effective rank via singular values of the centered matrix
    s = torch.linalg.svdvals(centered)
    s = s[s > 1e-12]
    if s.numel() == 0:
        erank = 1.0
    else:
        p = s / s.sum()
        entropy = -(p * p.log()).sum()
        erank = float(entropy.exp().item())

    inter_intra = None
    if labels is not None:
        lab = labels.detach().cpu().view(-1)
        intra_vals = []
        centroids = []
        for c in lab.unique():
            fc = f[lab == c]
            if fc.size(0) < 2:
                continue
            mu_c = fc.mean(dim=0, keepdim=True)
            centroids.append(mu_c.squeeze(0))
            intra_vals.append((fc - mu_c).pow(2).sum(dim=1).mean())
        if intra_vals:
            intra = torch.stack(intra_vals).mean()
            cmat = torch.stack(centroids)
            inter = (cmat - cmat.mean(dim=0, keepdim=True)).pow(2).sum(dim=1).mean()
            inter_intra = float((inter / (intra + 1e-12)).item())

    return {
        "erank": erank,
        "tvar": tvar,
        "inter_intra": inter_intra if inter_intra is not None else float("nan"),
        "n_samples": int(n),
    }
