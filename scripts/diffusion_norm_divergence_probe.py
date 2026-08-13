#!/usr/bin/env python3
"""量 diffusion 的「denoiser 有聚合 vs normalization buffer 沒聚合」在跨節點分歧上的落差。

【為什麼要量這個】
FOOGD 的 SAG 用 KSD 對齊，其 score model 是 **server 聚合後的全域模型**（原文：
"score models ... are aggregated into a global score model on the server, they inherit
distribution information from all participating clients"）。

MATCHA 的 KSD 耦合（train.py:1256-1261）：
    def _score_fn(z):
        z_n = diffusion_model.normalize(z)          # ← FeatureNormalization，全是 buffer
        return -diffusion_model.denoiser(z_n, t) / sigma_t

而 communicator.py:740-747 明確只聚合 named_parameters()：
    "Only aggregate denoiser parameters, NOT normalization buffers"
    "normalization buffers (mins, maxs, means, stds, etc.) are domain-specific statistics"

⇒ 各節點共用「同一個 denoiser」，卻把特徵餵進「各自不同的輸入座標系」。
本腳本量這個座標系到底差多少——若 denoiser 分歧 ≈ 0 而 normalization 分歧顯著，
即證實「全域 witness 被本地正規化打斷」，0803 §6 的 caveat 應據此改寫。

【指標】對齊 BN-DIV 定義（0730）：div = mean_i ||v_i − μ|| / ||μ||，μ = 跨節點平均。

純 checkpoint 讀取、零前向、零 GPU、不改任何檔案。

用法：
  venv_matcha/bin/python scripts/diffusion_norm_divergence_probe.py [RUN_DIR]
"""
import glob
import os
import re
import sys

import torch

ROOT = "/home/server5090/Desktop/M11307320/MATCHA"
DEFAULT_RUN = ("exp_result_v1_stage2_leave_cartoon_async_const_tau1e-5_style_"
               "osdg_excl_person_seed2026_topo1234")

# normalization buffer 的名字（dood/diffusion/diffusion_model.py:54 FeatureNormalization）
NORM_LEAVES = {"mins", "maxs", "means", "stds", "queue_ptr",
               "min", "max", "mean", "std", "shift", "scale"}


def divergence(tensors):
    """div = mean_i ||v_i - mu|| / ||mu||（與 0730 的 BN-DIV 同定義）。"""
    stack = torch.stack([t.double().flatten() for t in tensors])
    mu = stack.mean(0)
    mu_norm = mu.norm()
    if mu_norm < 1e-12:
        return float("nan")
    return float(((stack - mu).norm(dim=1) / mu_norm).mean())


def classify(key):
    """把 state_dict 的 key 分到四個桶。"""
    is_diff = key.startswith("diffusion_model.")
    leaf = key.split(".")[-1]
    if is_diff:
        # normalization 的 buffer 都掛在 diffusion_model.normalization.*
        if ".normalization." in key or leaf in NORM_LEAVES:
            return "diffusion normalization buffer（★ 不聚合）"
        return "diffusion denoiser 參數（有聚合）"
    if leaf in ("running_mean", "running_var"):
        return "backbone BN running（不聚合）"
    if leaf == "num_batches_tracked":
        return None
    return "backbone conv/fc 參數（有聚合）"


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN
    # 兩種命名並存：stage-2 九節點用 *_node_N_final.pth；stage-1 三節點用 *_<domain>_final.pth
    files = sorted(glob.glob(os.path.join(ROOT, run, "*_node_*_final.pth")),
                   key=lambda p: int(re.search(r"_node_(\d+)_final", p).group(1)))
    if not files:
        files = sorted(glob.glob(os.path.join(ROOT, run, "*_final.pth")))
    print(f"run: {run}")
    print(f"找到 {len(files)} 個節點 final checkpoint: "
          f"{[os.path.basename(f).replace('_final.pth', '').split('_')[-1] for f in files]}")
    if len(files) < 2:
        sys.exit("節點不足，無法量跨節點分歧")

    states = []
    for f in files:
        sd = torch.load(f, map_location="cpu", weights_only=False)
        for wrap in ("state_dict", "backbone_state"):
            if isinstance(sd, dict) and wrap in sd:
                sd = sd[wrap]
        states.append(sd)

    k0 = states[0]
    if not any(k.startswith("diffusion_model.") for k in k0):
        print("\n⚠️ 此 run 的 checkpoint 不含 diffusion_model.*（可能未開 --use_ood）。")
        print("   樣本 keys:", [k for k in list(k0)[:12]])
        sys.exit(1)

    buckets = {}
    for k in k0:
        bucket = classify(k)
        if bucket is None:
            continue
        if not torch.is_tensor(k0[k]) or k0[k].numel() == 0:
            continue
        if not all(k in s for s in states):
            continue
        try:
            div = divergence([s[k] for s in states])
        except RuntimeError:
            continue
        if div == div:  # 非 NaN
            buckets.setdefault(bucket, []).append((k, div, k0[k].numel()))

    order = ["backbone conv/fc 參數（有聚合）",
             "diffusion denoiser 參數（有聚合）",
             "backbone BN running（不聚合）",
             "diffusion normalization buffer（★ 不聚合）"]

    print(f"\n{'桶':<40} {'張量數':>7} {'div 中位':>10} {'div 最大':>10}")
    print("-" * 72)
    summary = {}
    for b in order:
        rows = buckets.get(b)
        if not rows:
            continue
        divs = sorted(r[1] for r in rows)
        med = divs[len(divs) // 2]
        summary[b] = med
        print(f"{b:<40} {len(rows):7d} {med:10.3e} {divs[-1]:10.3e}")
    print("-" * 72)

    print("\n逐張量明細 —— diffusion normalization buffer：")
    for k, div, n in sorted(buckets.get("diffusion normalization buffer（★ 不聚合）", []),
                            key=lambda r: -r[1]):
        print(f"  {k:<58} numel={n:<8} div={div:.4e}")

    agg = summary.get("diffusion denoiser 參數（有聚合）")
    nrm = summary.get("diffusion normalization buffer（★ 不聚合）")
    if agg is not None and nrm is not None and agg > 0:
        print(f"\n★ 判讀：denoiser 分歧中位 {agg:.3e}，normalization 分歧中位 {nrm:.3e}"
              f"（相差 {nrm / agg:.3g} 倍）。")
        print("   ⇒ 節點間共用同一個 denoiser，但把特徵餵進各自不同的輸入座標系。")


if __name__ == "__main__":
    main()
