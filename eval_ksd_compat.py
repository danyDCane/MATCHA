"""Static target-vs-clean KSD distribution-compatibility eval (KSD-on vs V1).

Tests whether KSD made domain-shifted (target=cartoon) features relatively MORE
compatible with each model's OWN clean-source feature density — avoiding (a) z_aug
style-shift reconstruction and (b) cross-feature-space comparison (we use a
within-model ratio under each model's own frozen diffusion).

Per source node N (art/photo/sketch), per version (KSD-on / V1):
  - clean = domain N test features ; target = cartoon test features ; tex = DTD textures features
    (all via backbone.intermediate_forward = eval, no style-shift -> deterministic, no reconstruction)
  - q_N = node N's OWN frozen diffusion (the clean-density model in N's feature space)
  - bw = median-heuristic of the clean set (SAME ruler for clean/target/tex)
  - KSD(set ‖ q_N) at t in {10,25,50}, bootstrapped (reps x n) -> mean±std
  - ratio_target = KSD_target/KSD_clean (primary, scale-normalized, comparable across versions)
    ratio_tex = KSD_tex/KSD_clean (sanity: should be >>1 if the KSD measure detects off-manifold)
  - eps_mse mean per set (density-proxy sanity gap)
  - target intrinsic linear separability (class-info concern: KSD is class-agnostic)

Run:
  venv_matcha/bin/python eval_ksd_compat.py
"""
import os
import types
import numpy as np
import torch

import util
from test_domain_ood_scores import (
    load_checkpoint, load_pacs_test_data, load_textures_ood_loader, resolve_textures_root,
)
from dood.utils.diffusion import get_diffusion_model, get_diffusion_scores
from dood.ksd import compute_KSD, SE_kernel_multi, trace_SE_kernel_multi, median_heruistic

DATASET_ROOT = "/home/server5090/Desktop/M11307320/datasets"
DEVICE = "cuda"
TARGET = "cartoon"               # leave-out / held-out target domain (cartoon LOO)
SOURCES = ["art_painting", "photo", "sketch"]
T_LIST = [10, 25, 50]
KSD_N = 256                      # samples per KSD U-statistic
KSD_REPS = 5                    # bootstrap repetitions
SEED = 1234
VERSIONS = {
    "control":  ("exp_result_v1_stage1_leave_cartoon_ksd_r0ctrl_t25", "v1_stage1_leave_cartoon_ksd_r0ctrl_t25"),
    "KSD-r0.3": ("exp_result_v1_stage1_leave_cartoon_ksd_r0.3_t25",   "v1_stage1_leave_cartoon_ksd_r0.3_t25"),
    "KSD-r1.0": ("exp_result_v1_stage1_leave_cartoon_ksd_r1.0_t25",   "v1_stage1_leave_cartoon_ksd_r1.0_t25"),
}


def _build_args():
    a = types.SimpleNamespace()
    a.model = "res"; a.dataset = "pacs"; a.resnet_type = "standard"; a.pretrained = False
    a.num_classes = 7
    return a


@torch.no_grad()
def extract_features(backbone, loader, max_n):
    backbone.eval()
    feats, labels = [], []
    for batch in loader:
        data, label, _ = util.unpack_batch(batch)
        z = backbone.intermediate_forward(data.to(DEVICE))
        feats.append(z.detach().float().cpu())
        labels.append(torch.as_tensor(label).cpu())
        if sum(f.shape[0] for f in feats) >= max_n:
            break
    return torch.cat(feats, 0)[:max_n], torch.cat(labels, 0)[:max_n]


def make_score_fn(diffusion_model, t):
    dp = diffusion_model.diffusion_process
    t = max(1, min(int(t), int(dp.num_timesteps) - 1))
    sig = float(dp.sqrt_one_minus_alphas_cumprod[t])

    def score_fn(z):
        with torch.no_grad():  # detached score values are enough for the KSD VALUE
            zn = diffusion_model.normalize(z)
            tt = torch.full((z.size(0),), t, device=z.device, dtype=torch.long)
            return -diffusion_model.denoiser(zn, tt) / (sig + 1e-8)
    return score_fn


@torch.no_grad()
def score_dir_to_clean(diffusion_model, score_fn, tgt_f, clean_f):
    """Does the frozen diffusion score at TARGET point back toward the clean-source manifold?
    Mirrors Phase 0 score_diag but on held-out target (no paired clean -> clean centroid ref).
    Returns (cos_to_clean_mean, frac_inward, |score(tgt)|, |score(clean)|).
    cos>0 / frac>0.5 = score points target back toward clean (witness usable at target's distance)."""
    zt = diffusion_model.normalize(tgt_f.to(DEVICE).float())
    zc = diffusion_model.normalize(clean_f.to(DEVICE).float())
    disp = zc.mean(dim=0, keepdim=True) - zt          # per-target direction toward clean centroid
    s_t = score_fn(tgt_f.to(DEVICE).float())          # score at target (normalized space)
    s_c = score_fn(clean_f.to(DEVICE).float())
    cos = (s_t * disp).sum(1) / (s_t.norm(dim=1) * disp.norm(dim=1) + 1e-8)
    return (float(cos.mean()), float((cos > 0).float().mean()),
            float(s_t.norm(dim=1).mean()), float(s_c.norm(dim=1).mean()))


def ksd_value(feat_subset, score_fn, bw):
    z = feat_subset.to(DEVICE).float().requires_grad_(True)  # require grad for kernel-grad autograd
    val = compute_KSD(z, z, score_fn, SE_kernel_multi, trace_SE_kernel_multi, bw,
                      flag_U=True, flag_retain=True, flag_create=False)
    return float(val.item())


def bootstrap_ksd(feat, score_fn, bw, gen):
    n = min(KSD_N, feat.shape[0])
    vals = [ksd_value(feat[torch.randperm(feat.shape[0], generator=gen)[:n]], score_fn, bw)
            for _ in range(KSD_REPS)]
    return float(np.mean(vals)), float(np.std(vals))


def eps_mse_mean(diffusion_model, feat):
    try:
        scores, _ = get_diffusion_scores(feat.to(DEVICE), diffusion_model, list(range(25)), "eps_mse",
                                         normalize=True, dtype=torch.float32)
        s = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
        return float(np.mean(s))
    except Exception as e:
        print(f"[eps_mse][WARN] {e}", flush=True)
        return float("nan")


def linear_probe_acc(feat, labels):
    """Intrinsic class-separability of target features (70/30 split)."""
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return float("nan")
    X = feat.numpy(); y = labels.numpy()
    g = np.random.RandomState(SEED); idx = g.permutation(len(y)); cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X[tr], y[tr])
    return float((clf.predict(X[te]) == y[te]).mean())


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    tex_root = resolve_textures_root(DATASET_ROOT, os.path.join(DATASET_ROOT, "dtd/images"))
    args = _build_args()
    rows = []

    for vname, (ckpt_dir, desc) in VERSIONS.items():
        backbone = util.select_model(args.num_classes, args).to(DEVICE)
        diffusion_model = get_diffusion_model(ft_size=512, denoiser_type="unet0d",
                                              diffusion_denoiser_channels=512,
                                              num_diffusion_steps=1000).to(DEVICE)
        for node in SOURCES:
            ckpt_path = os.path.abspath(os.path.join(ckpt_dir, f"{desc}_{node}_final.pth"))
            if not os.path.exists(ckpt_path):
                print(f"[WARN] missing ckpt: {ckpt_path}"); continue
            load_checkpoint(ckpt_path, backbone, diffusion_model, DEVICE)
            diffusion_model.eval()

            clean_f, clean_y = extract_features(backbone, load_pacs_test_data(DATASET_ROOT, node, 128, 4)[0], 1024)
            tgt_f, tgt_y     = extract_features(backbone, load_pacs_test_data(DATASET_ROOT, TARGET, 128, 4)[0], 1024)
            tex_f, _         = extract_features(backbone, load_textures_ood_loader(DATASET_ROOT, tex_root, 128, 4)[0], 1024)

            bw = median_heruistic(clean_f[:KSD_N].to(DEVICE).float(), clean_f[:KSD_N].to(DEVICE).float())
            probe = linear_probe_acc(tgt_f, tgt_y)
            for t in T_LIST:
                sfn = make_score_fn(diffusion_model, t)
                cdir, finw, snt, snc = score_dir_to_clean(diffusion_model, sfn, tgt_f, clean_f)
                print(f"[score_dir] {vname} {node} t={t} cos_to_clean={cdir:+.4f} "
                      f"frac_inward={finw:.3f} |s_tgt|={snt:.4f} |s_clean|={snc:.4f} "
                      f"(ratio={snt/(snc+1e-8):.3f})", flush=True)
                gen = torch.Generator().manual_seed(SEED)
                kc, kc_s = bootstrap_ksd(clean_f, sfn, bw, gen)
                kt, kt_s = bootstrap_ksd(tgt_f, sfn, bw, gen)
                kx, kx_s = bootstrap_ksd(tex_f, sfn, bw, gen)
                rows.append(dict(version=vname, node=node, t=t, bw=float(bw),
                                 KSD_clean=kc, KSD_clean_std=kc_s, KSD_target=kt, KSD_target_std=kt_s,
                                 KSD_tex=kx, ratio_target=kt / (kc + 1e-12), ratio_tex=kx / (kc + 1e-12),
                                 eps_clean=eps_mse_mean(diffusion_model, clean_f),
                                 eps_target=eps_mse_mean(diffusion_model, tgt_f),
                                 probe_target=probe))
                print(f"[{vname} {node} t={t}] KSD_clean={kc:.3f}±{kc_s:.2f} KSD_target={kt:.3f}±{kt_s:.2f} "
                      f"ratio_target={kt/(kc+1e-12):.3f} | KSD_tex={kx:.2f} ratio_tex={kx/(kc+1e-12):.2f} "
                      f"| probe_target={probe:.3f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/ksd_compat_cartoon.csv", index=False)
    print("\n================ SUMMARY (avg over 3 source nodes) ================")
    for t in T_LIST:
        print(f"\n--- t={t} ---")
        for vname in VERSIONS:
            sub = df[(df.version == vname) & (df.t == t)]
            if len(sub) == 0:
                continue
            print(f"  {vname:7s}: ratio_target={sub.ratio_target.mean():.3f}  "
                  f"ratio_tex(sanity)={sub.ratio_tex.mean():.2f}  "
                  f"KSD_clean={sub.KSD_clean.mean():.3f}  "
                  f"eps_gap(tgt-clean)={(sub.eps_target.mean()-sub.eps_clean.mean()):+.3f}  "
                  f"probe_target={sub.probe_target.mean():.3f}")
    print("\nSaved: results/ksd_compat_cartoon.csv")


if __name__ == "__main__":
    main()
