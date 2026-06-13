"""Kernelized Stein Discrepancy (KSD) — pure functions for the KSD generalization
coupling (Phase 1).

Ported verbatim (behaviour-preserving) from FOOGD
``src/algorithms/FOOGD/ksd.py`` — only the four functions actually used by the
coupling are kept: ``median_heruistic``, ``SE_kernel_multi``,
``trace_SE_kernel_multi`` and ``compute_KSD``. Source: "Sliced Kernelized Stein
Discrepancy".

Usage in MATCHA coupling (train.py), with z_aug = grad-enabled style-shifted
penultimate and a frozen-diffusion score witness ``score_fn``::

    bw = median_heruistic(z_aug.detach(), z_aug.detach())
    loss_ksd = compute_KSD(z_aug, z_aug, score_fn,
                           SE_kernel_multi, trace_SE_kernel_multi, bw,
                           flag_U=True, flag_retain=True, flag_create=True)

CRITICAL (the make-or-break detail): Term2/Term3 below take kernel gradients via
``torch.autograd.grad(..., create_graph=flag_create, retain_graph=flag_retain)``.
The FOOGD defaults are ``False`` — with the defaults the KSD value is detached
from the kernel-gradient terms, so ``loss_ksd.backward()`` delivers NO gradient
through them to the backbone. For the coupling you MUST pass
``flag_retain=True, flag_create=True`` (and ``flag_U=True`` for the U-statistic).
"""

import torch


def trace_SE_kernel_multi(sample1, sample2, bandwidth, K):
    """Trace of the 2nd-order derivative of the RBF kernel.

    sample1, sample2: ... x N x N x dim ; K: ... x N x N -> ... x N x N
    """
    diff = sample1 - sample2  # N x N x dim
    H = K * (2. / (bandwidth ** 2 + 1e-9) * sample1.shape[-1]
             - 4. / (bandwidth ** 4 + 1e-9) * torch.sum(diff * diff, dim=-1))  # N x N
    return H


def SE_kernel_multi(sample1, sample2, bandwidth):
    """Multidim squared-exponential (RBF) kernel.

    sample1, sample2: ... x N x N x dim -> ... x N x N
    """
    if len(sample1.shape) == 4:  # * x N x N x d
        bandwidth = bandwidth.unsqueeze(-1).unsqueeze(-1)

    sample_diff = sample1 - sample2  # N x N x dim
    # squared L2 == ||diff||^2 ; use (diff**2).sum (NOT torch.norm()**2) so the kernel stays
    # smooth at diff==0. The MATCHA coupling runs the U-statistic on (z_aug, z_aug) (same set),
    # whose diagonal has diff==0 where torch.norm's 2nd-order grad is NaN under create_graph=True.
    norm_sample = (sample_diff ** 2).sum(dim=-1)  # N x N or * x N x N
    K = torch.exp(-norm_sample / (bandwidth ** 2 + 1e-9))
    return K


def median_heruistic(sample1, sample2):
    """Median-heuristic bandwidth (detached scalar / per-* vector).

    `dist` is the SQUARED pairwise distance, so its median is median(||x-y||^2).
    We return sqrt(median) so that the caller's `bandwidth**2` denominator equals
    median(||x-y||^2) — the textbook RBF lengthscale K=exp(-d^2/median(d^2)).
    (FOOGD returns the squared-distance median directly; combined with the bw**2
    denominator in SE_kernel that double-squares the scale and gives a near-flat
    kernel unless features are pre-normalized — see ksd smoke.)
    """
    with torch.no_grad():
        G = torch.sum(sample1 * sample1, dim=-1)  # N or * x N
        G_exp = torch.unsqueeze(G, dim=-2)  # 1 x N or * x 1 x N

        H = torch.sum(sample2 * sample2, dim=-1)
        H_exp = torch.unsqueeze(H, dim=-1)  # N x 1 or * x N x 1
        dist = G_exp + H_exp - 2 * sample2.matmul(torch.transpose(sample1, -1, -2))  # squared dist
        if len(dist.shape) == 3:
            dist = dist[torch.triu(torch.ones(dist.shape)) == 1].view(dist.shape[0], -1)  # * x (NN)
            median_dist, _ = torch.median(dist, dim=-1)  # *
        else:
            dist = (dist - torch.tril(dist)).view(-1)
            median_dist = torch.median(dist[dist > 0.])
        median_dist = torch.sqrt(torch.clamp(median_dist, min=1e-12))
    return median_dist.clone().detach()


def compute_KSD(samples1, samples2, score_func, kernel, trace_kernel, bandwidth,
                score_sample1=None, score_sample2=None,
                flag_U=False, flag_retain=False, flag_create=False):
    """Kernelized Stein Discrepancy U-/V-statistic.

    score_func models \\nabla_x log p_\\theta(x). For the MATCHA coupling pass
    flag_U=True, flag_retain=True, flag_create=True so the result stays
    differentiable w.r.t. samples (=> backbone) through the kernel-gradient terms.
    """
    divergence_accum = 0

    samples1_crop_exp = torch.unsqueeze(samples1, dim=1).repeat(1, samples2.shape[0], 1)  # N x N(rep) x dim
    samples2_crop_exp = torch.unsqueeze(samples2, dim=0).repeat(samples1.shape[0], 1, 1)  # N(rep) x N x dim

    # Term 1
    if (score_sample1 is None) or (score_sample2 is None):
        score_sample1 = score_func(samples1)  # N x dim
        score_sample2 = score_func(samples2)  # N x dim

    score_sample1_exp = torch.unsqueeze(score_sample1, dim=1)  # N x 1 x dim
    score_sample2_exp = torch.unsqueeze(score_sample2, dim=0)  # 1 x N x dim

    K = kernel(samples1_crop_exp, samples2_crop_exp, bandwidth=bandwidth)

    if flag_U:
        Term1 = (K - torch.diag(torch.diag(K))) * torch.sum(score_sample1_exp * score_sample2_exp, dim=-1)  # N x N
    else:
        Term1 = (K) * torch.sum(score_sample1_exp * score_sample2_exp, dim=-1)  # N x N

    # Term 2 — kernel gradient w.r.t. samples2 (autograd)
    if flag_U:
        grad_K_2 = torch.autograd.grad(
            torch.sum((K - torch.diag(torch.diag(K)))), samples2_crop_exp,
            retain_graph=flag_retain, create_graph=flag_create)[0]  # N x N x dim
    else:
        grad_K_2 = torch.autograd.grad(
            torch.sum((K)), samples2_crop_exp,
            retain_graph=flag_retain, create_graph=flag_create)[0]  # N x N x dim
    Term2 = torch.sum(score_sample1_exp * grad_K_2, dim=-1)  # N x N

    # Term 3 — kernel gradient w.r.t. samples1 (autograd)
    if flag_U:
        K = kernel(samples1_crop_exp, samples2_crop_exp, bandwidth=bandwidth)
        grad_K_1 = torch.autograd.grad(
            torch.sum((K - torch.diag(torch.diag(K)))), samples1_crop_exp,
            retain_graph=flag_retain, create_graph=flag_create)[0]  # N x N x dim
    else:
        K = kernel(samples1_crop_exp, samples2_crop_exp, bandwidth=bandwidth)
        grad_K_1 = torch.autograd.grad(
            torch.sum((K)), samples1_crop_exp,
            retain_graph=flag_retain, create_graph=flag_create)[0]  # N x N x dim
    Term3 = torch.sum(score_sample2_exp * grad_K_1, dim=-1)  # N x N

    # Term 4 — trace of high-order derivative of kernel (manual)
    K = kernel(samples1_crop_exp, samples2_crop_exp, bandwidth=bandwidth)
    if flag_U:
        T_K = trace_kernel(samples1_crop_exp, samples2_crop_exp, bandwidth=bandwidth, K=K - torch.diag(torch.diag(K)))
    else:
        T_K = trace_kernel(samples1_crop_exp, samples2_crop_exp, bandwidth=bandwidth, K=K)
    Term4 = T_K  # N x N

    KSD_comp = torch.sum(Term1 + 1 * Term2 + 1 * Term3 + 1 * Term4)
    divergence_accum += KSD_comp

    if flag_U:
        KSD = divergence_accum / ((samples1.shape[0] - 1) * samples2.shape[0])
    else:
        KSD = divergence_accum / (samples1.shape[0] * samples2.shape[0])

    return KSD
