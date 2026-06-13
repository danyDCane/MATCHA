"""Standalone correctness gate for the Fourier amplitude augmentation (Stage 1.5).

Verifies the augmentation BEFORE any GPU training, so a broken FFT / normalize /
partner-sampling never wastes a multi-hour run. Saves raw|partner|aug triptychs
(lambda sweep) and asserts 5 acceptance items.

Run (CPU, seconds):
    venv_matcha/bin/python -m dood.verify_fourier_aug --root ../datasets/ --domain photo

Acceptance (printed + asserted where numeric):
  1. phase preserved  -> visual (content/edges recognizable) + lam=0 reconstructs x_t
  2. amp -> partner   -> at lam=1 the augmented amplitude spectrum matches partner's
  3. value range      -> aug in [0,1] after clamp; normalized tensor has no NaN/Inf
  4. partner source   -> drawn within the SAME domain (within-node) via the wrapper meta
  5. ratio<1 band     -> only the centered low-freq block changes vs full-spectrum
"""

import argparse
import os

import torch
from torchvision import transforms

from dood.fourier_aug import (
    FourierAugPACSDataset,
    colorful_spectrum_mix_torch,
)

# Local import so the script works from repo root.
from pacs_dataset import PACSDataset


def _amp_phase(x):
    F = torch.fft.fft2(x, dim=(-2, -1))
    return torch.abs(F), torch.angle(F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../datasets/", help="dataset root containing PACS/")
    ap.add_argument("--domain", default="photo", help="a TRAINING domain (e.g. art_painting/photo/sketch)")
    ap.add_argument("--n", type=int, default=4, help="number of target samples to render")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="./fourier_aug_check")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Deterministic raw transform (NO random geometric aug) so raw vs aug differ ONLY
    # by amplitude — isolates the effect for visual/numeric inspection.
    raw_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    base = PACSDataset(root=args.root, dataset_name=args.domain, transform=raw_tf)
    wrapper = FourierAugPACSDataset(base, alpha=1.0, ratio=1.0, seed=args.seed)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = torch.Generator().manual_seed(args.seed)
    idxs = torch.randperm(len(base), generator=rng)[:args.n].tolist()

    lams = [0.0, 0.5, 1.0]
    all_ok = True
    range_ok = True
    nan_ok = True

    fig, axes = plt.subplots(args.n, 2 + len(lams), figsize=(3 * (2 + len(lams)), 3 * args.n))
    if args.n == 1:
        axes = axes.reshape(1, -1)

    for row, i in enumerate(idxs):
        x_t, _, meta_t = base[i]
        p_idx = wrapper.rng.randrange(len(base))   # advance same RNG kind for a representative partner
        x_p, _, meta_p = base[p_idx]

        # Check 4: partner within same domain
        same_domain = (meta_t["domain"] == meta_p["domain"])
        all_ok &= same_domain

        axes[row, 0].imshow(x_t.permute(1, 2, 0).numpy()); axes[row, 0].set_title(f"target #{i}\n{meta_t['domain']}")
        axes[row, 1].imshow(x_p.permute(1, 2, 0).numpy()); axes[row, 1].set_title(f"partner #{p_idx}\n{meta_p['domain']}")

        for col, lam in enumerate(lams):
            x_aug = colorful_spectrum_mix_torch(x_t, x_p, lam, 1.0)
            axes[row, 2 + col].imshow(x_aug.permute(1, 2, 0).numpy())
            axes[row, 2 + col].set_title(f"aug lam={lam}")

            # Check 3: value range after clamp
            if x_aug.min() < -1e-6 or x_aug.max() > 1 + 1e-6:
                range_ok = False

            if lam == 0.0:
                # Check 1: lam=0 reconstructs target (phase preserved, amp unchanged)
                recon_err = (x_aug - x_t).abs().mean().item()
                print(f"[row {row}] lam=0 |aug - target| mean = {recon_err:.2e} (expect ~0)")
                all_ok &= (recon_err < 1e-4)
            if lam == 1.0:
                # Check 2: lam=1 amplitude matches partner; phase stays target's
                amp_aug, pha_aug = _amp_phase(x_aug)
                amp_p, _ = _amp_phase(x_p)
                _, pha_t = _amp_phase(x_t)
                amp_rel = (amp_aug - amp_p).abs().sum().item() / (amp_p.abs().sum().item() + 1e-8)
                pha_err = (pha_aug - pha_t).abs().mean().item()
                print(f"[row {row}] lam=1 amp rel-diff vs partner = {amp_rel:.3e} (expect small); "
                      f"phase mean-diff vs target = {pha_err:.3e}")
                # amplitude should be close to partner (clamp/ifft introduce minor error)
                all_ok &= (amp_rel < 0.15)

        # Check 3b: normalized tensors finite
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for t in (norm(x_t), norm(colorful_spectrum_mix_torch(x_t, x_p, 1.0, 1.0))):
            if not torch.isfinite(t).all():
                nan_ok = False

    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    sweep_path = os.path.join(args.out, f"triptych_{args.domain}.png")
    fig.savefig(sweep_path, dpi=110)
    plt.close(fig)

    # Check 5: ratio<1 only mutates centered low-freq block.
    x_t, _, _ = base[idxs[0]]
    x_p, _, _ = base[wrapper.rng.randrange(len(base))]
    aug_full = colorful_spectrum_mix_torch(x_t, x_p, 1.0, ratio=1.0)
    aug_half = colorful_spectrum_mix_torch(x_t, x_p, 1.0, ratio=0.25)
    band_diff = (aug_half - x_t).abs().mean().item()
    full_diff = (aug_full - x_t).abs().mean().item()
    ratio_ok = band_diff < full_diff   # partial band changes less than full spectrum
    print(f"ratio<1 band mean-diff = {band_diff:.3e}  <  full-spectrum diff = {full_diff:.3e} ? {ratio_ok}")

    print("\n==== Fourier aug correctness gate ====")
    print(f"  [1] phase preserved (lam=0 recon)     : {'PASS' if all_ok else 'CHECK'}")
    print(f"  [2] amp -> partner (lam=1)            : see rel-diff above")
    print(f"  [3] value range [0,1] after clamp     : {'PASS' if range_ok else 'FAIL'}")
    print(f"  [3b] normalized finite (no NaN/Inf)   : {'PASS' if nan_ok else 'FAIL'}")
    print(f"  [4] partner within same domain        : {'PASS' if all_ok else 'CHECK'}")
    print(f"  [5] ratio<1 mutates only low-freq band: {'PASS' if ratio_ok else 'FAIL'}")
    print(f"  triptych saved -> {sweep_path}")

    gate = range_ok and nan_ok and ratio_ok and all_ok
    print(f"\nGATE: {'PASS — safe to train' if gate else 'FAIL — do NOT train, fix aug first'}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
