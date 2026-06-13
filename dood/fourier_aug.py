"""Image-level Fourier amplitude augmentation for MATCHA (Option B, Path F).

Within-node partner amplitude swap (phase preserved), torch implementation.
Adapted from FOOGD `colorful_spectrum_mix` (numpy, src/data/data_utils.py) and the
torch `_pair_amp_swap` from the `evaluate_ood` branch probe script.

Design notes (see research/V2B1_score_norm/0601_fourier_aug_within_node_cartoon_plan.md):
  * The wrapper returns a 4-tuple ``(img_raw_norm, img_aug_norm, target, meta)``.
    - img_raw_norm : original image, ImageNet-normalized   -> Path S input
    - img_aug_norm : amplitude-augmented image, normalized  -> Path F input
  * Partner is drawn within the SAME node/domain (vanilla FOOGD correspondence).
  * Partner index and lambda are sampled from a dataset-local ``random.Random``
    instance (NOT the global RNG) so the global torch/random streams stay
    bit-identical to the baseline run -> single-variable A/B (only Fourier on/off).
    With ``num_workers=0`` (MATCHA train loader default) this is fully reproducible
    and still varies across epochs. For ``num_workers>0`` use ``fourier_worker_init_fn``.
"""

import math
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def colorful_spectrum_mix_torch(x_t, x_p, lam, ratio=1.0):
    """Amplitude-swap Fourier augmentation, phase of ``x_t`` preserved.

    Args:
        x_t: target image, ``[C,H,W]`` or ``[B,C,H,W]`` float in [0,1].
        x_p: partner image, same shape as ``x_t``, float in [0,1].
        lam: amplitude mixing coefficient in [0,1] (fraction taken from partner).
        ratio: fraction of the centered low-frequency band to mix; 1.0 = full spectrum.

    Returns:
        Augmented image, same shape as ``x_t``, clamped to [0,1].
    """
    H, W = x_t.shape[-2:]
    F_t = torch.fft.fft2(x_t, dim=(-2, -1))
    F_p = torch.fft.fft2(x_p, dim=(-2, -1))
    abs_t, ang_t = torch.abs(F_t), torch.angle(F_t)
    abs_p = torch.abs(F_p)
    abs_t_s = torch.fft.fftshift(abs_t, dim=(-2, -1))
    abs_p_s = torch.fft.fftshift(abs_p, dim=(-2, -1))

    if ratio >= 1.0:
        abs_new_s = (1.0 - lam) * abs_t_s + lam * abs_p_s
    else:
        h_crop = int(H * math.sqrt(ratio))
        w_crop = int(W * math.sqrt(ratio))
        h_st = H // 2 - h_crop // 2
        w_st = W // 2 - w_crop // 2
        abs_new_s = abs_t_s.clone()
        abs_new_s[..., h_st:h_st + h_crop, w_st:w_st + w_crop] = (
            (1.0 - lam) * abs_t_s[..., h_st:h_st + h_crop, w_st:w_st + w_crop]
            + lam * abs_p_s[..., h_st:h_st + h_crop, w_st:w_st + w_crop]
        )

    abs_new = torch.fft.ifftshift(abs_new_s, dim=(-2, -1))
    F_new = abs_new * torch.exp(1j * ang_t)
    x_new = torch.real(torch.fft.ifft2(F_new, dim=(-2, -1)))
    return x_new.clamp(0.0, 1.0)


class FourierAugPACSDataset(Dataset):
    """Option B wrapper: returns (raw_norm, fourier_aug_norm, target, meta).

    Args:
        base_dataset: a ``PACSDataset`` built with a RAW transform
            (Resize + ToTensor, NO Normalize), so its items are [3,H,W] in [0,1].
        alpha: lambda ~ U(0, alpha); FOOGD default 1.0.
        ratio: low-frequency band fraction to mix (1.0 = full spectrum).
        seed: seed for the dataset-local partner/lambda RNG.
    """

    def __init__(self, base_dataset, alpha=1.0, ratio=1.0, seed=1234):
        self.base = base_dataset
        self.alpha = float(alpha)
        self.ratio = float(ratio)
        self.seed = int(seed)
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self._n = len(self.base)
        # Independent RNG so the global torch/random streams are untouched.
        self.rng = random.Random(self.seed)

    def __len__(self):
        return self._n

    def __getitem__(self, index):
        x_ori, target, meta = self.base[index]          # [3,H,W] in [0,1]
        p_idx = self.rng.randrange(self._n)             # within-node partner
        x_p, _, _ = self.base[p_idx]
        lam = self.rng.uniform(0.0, self.alpha)

        x_aug = colorful_spectrum_mix_torch(x_ori, x_p, lam, self.ratio)

        img_raw_norm = self.normalize(x_ori)
        img_aug_norm = self.normalize(x_aug)

        meta = dict(meta)
        meta["partner_index"] = p_idx
        meta["fourier_lam"] = float(lam)
        return img_raw_norm, img_aug_norm, target, meta


def fourier_worker_init_fn(worker_id):
    """Reseed dataset-local RNG per worker for ``num_workers>0`` reproducibility.

    Not needed for the default MATCHA train loader (``num_workers=0``); provided
    as a fallback if workers are ever enabled.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    if isinstance(ds, FourierAugPACSDataset):
        ds.rng = random.Random(ds.seed + worker_id)
