"""0629 好鄰居 BN-running 訊號 gating (Stage 2).

Plan: research/bn_fusion/0629_stage2_good_neighbor_bn_signal_gating_plan.md

問題（單一）：是否存在「不看 target 的免費訊號 S_j」，其對 9 個節點 BN-running 的
排序顯著預測真實 target acc A_j、且 argmax 逼近 oracle headroom？

關鍵事實（已釘死、見 plan §2/§3）：
  - best-single = global broadcast：廣播節點 j 的 BN 給全體 ≡ conv_shared+BN_j = 節點 j
    自己的 baseline → A_j = 節點 j 在 target 上的 baseline acc（直接 eval 節點 j checkpoint）。
  - 節點 = 1 源域 × 3 複本（9 BN 塌縮成 3 域群）。node→domain 由 all_domains 固定序
    contiguous 重建（util.assign_nodes_to_domains 等價），並以「自源域 acc 最高」+ R4 聚類驗證。
  - conv/affine/fc 跨節點 1e-7 同質；只有 BN-running 分歧 → 用節點 j 自己的 model 即可。

純 post-hoc inference、零訓練、不改任何訓練/聚合 code、不覆寫 checkpoint。

訊號：
  Family 1（幾何、零 forward，從 9 個 BN 向量算）：
    S_centroid = -dist(BN_j, mean_k BN_k)      （質心近＝代表性）
    S_prof     = mean_{i!=j} dist(BN_j, BN_i)   （教授：互距大＝權重大）
    S_varmag   = mean running_var               （變異量級）
    距離 × 層集合：{L2_std, W2} × {early(layer1/2/3), all}
  Family 2（跨源域 transfer、gating 允許 forward）：
    S_xdom_acc = 節點 j 的 model 在「其他源域」上的分類 acc（有標籤）
    S_xdom_negent = -（其他源域上的平均預測熵）（label-free）

用法：
  venv_matcha/bin/python scripts/bn_signal_gating.py \
    --datasetRoot ../datasets/ \
    --checkpoint_root . \
    --folds art_painting,cartoon,photo,sketch
"""

import argparse
import csv
import glob
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

import util
from test_domain_ood_scores import load_pacs_test_data

ALL_DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
EARLY_LAYER_TAGS = ("layer1.", "layer2.", "layer3.")
# checkpoint dir/file token differs from PACS domain name for art_painting
DOMAIN_TO_CKPT_TOKEN = {"art_painting": "art"}


# ----------------------------- node/domain map -----------------------------

def reconstruct_node_to_domain(leave_out, num_nodes):
    """Contiguous assignment matching util.assign_nodes_to_domains
    (available_domains = all_domains order minus leave_out)."""
    available = [d for d in ALL_DOMAINS if d != leave_out]
    base, extra = divmod(num_nodes, len(available))
    nd, idx = {}, 0
    for i, dom in enumerate(available):
        for _ in range(base + (1 if i < extra else 0)):
            nd[f"node_{idx}"] = dom
            idx += 1
    return nd, available


# ----------------------------- model / eval -----------------------------

def build_backbone(ckpt_args, num_class, device):
    # pretrained irrelevant: weights are overwritten by checkpoint load
    setattr(ckpt_args, "pretrained", False)
    model = util.select_model(num_class, ckpt_args)
    return model.to(device)


def load_backbone_only(path, backbone, device):
    """Load backbone weights+buffers from a MATCHA checkpoint, skip diffusion
    (mirrors test_domain_ood_scores.load_checkpoint backbone branch; avoids its
    unconditional diffusion_model.load_state_dict when diffusion_model is None)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    state = {k: v for k, v in ck["backbone_state"].items()
             if not k.startswith("diffusion_model.")}
    backbone.load_state_dict(state, strict=False)  # keys keep 'backbone.' prefix (StandardResNetWrapper)
    return ck


@torch.no_grad()
def eval_acc(model, loader):
    """Top-1 acc (%) — identical path to util.test (model(x) in eval, no style aug)."""
    return float(util.test(model, loader))


@torch.no_grad()
def eval_mean_entropy(model, loader, device):
    """Mean predictive entropy (nats) over a loader."""
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        x, _, _ = util.unpack_batch(batch)
        x = x.cuda(non_blocking=True)
        logits = model(x)
        p = F.softmax(logits, dim=1)
        ent = -(p * torch.log(p + 1e-12)).sum(dim=1)
        tot += float(ent.sum().item())
        n += x.size(0)
    return tot / max(n, 1)


# ----------------------------- BN extraction -----------------------------

def extract_bn_vectors(model, layerset):
    """Return (mu, var) 1D np arrays concatenated over selected BN layers.
    layerset: 'early' (layer1/2/3) or 'all'."""
    mus, vars_ = [], []
    for name, buf in model.named_buffers():
        if name.endswith("running_mean"):
            base = name[: -len("running_mean")]
            if layerset == "early" and not any(t in name for t in EARLY_LAYER_TAGS):
                continue
            var_buf = dict(model.named_buffers())[base + "running_var"]
            mus.append(buf.detach().cpu().numpy().ravel())
            vars_.append(var_buf.detach().cpu().numpy().ravel())
    return np.concatenate(mus), np.concatenate(vars_)


# ----------------------------- distances / signals -----------------------------

def dist_L2_std(reps):
    """reps: [N, 2D] standardized per-dim. Returns NxN euclidean distance."""
    z = (reps - reps.mean(0, keepdims=True)) / (reps.std(0, keepdims=True) + 1e-8)
    diff = z[:, None, :] - z[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def dist_W2(mus, vars_):
    """Per-channel 2-Wasserstein between Gaussians N(mu,var): W2^2 = (dmu)^2+(dsig)^2.
    mus/vars_: [N, D]. Returns NxN."""
    sig = np.sqrt(np.clip(vars_, 0, None))
    dmu = mus[:, None, :] - mus[None, :, :]
    dsig = sig[:, None, :] - sig[None, :, :]
    return np.sqrt((dmu ** 2 + dsig ** 2).sum(-1))


def geometry_signals(D):
    """Given NxN distance matrix, return centroid-closeness and prof mutual-distance."""
    n = D.shape[0]
    prof = D.sum(1) / (n - 1)                       # mean dist to others (higher = professor's pick)
    # centroid closeness: -mean dist to all (proxy for centrality); higher = more central
    centroid = -D.mean(1)
    return centroid, prof


# ----------------------------- spearman -----------------------------

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


# ----------------------------- per-fold driver -----------------------------

def run_fold(target, args, device, out_rows, summary_rows):
    token = DOMAIN_TO_CKPT_TOKEN.get(target, target)
    ckpt_dir = os.path.join(args.checkpoint_root, f"exp_result_v1_stage2_leave_{token}_det")
    node_files = {}
    for j in range(args.num_nodes):
        hits = glob.glob(os.path.join(ckpt_dir, f"*node_{j}_final*.pth"))
        if not hits:
            raise FileNotFoundError(f"missing node {j} final ckpt in {ckpt_dir}")
        node_files[j] = sorted(hits)[0]

    node_to_domain, available = reconstruct_node_to_domain(target, args.num_nodes)
    print(f"\n===== FOLD leave-{target} =====")
    print(f"available source domains (order): {available}")
    print(f"node->domain: {node_to_domain}")

    # loaders: target (for A_j) + each source domain (for cross-source + sanity)
    target_loader, _ = load_pacs_test_data(args.datasetRoot, target, args.batch_size, args.num_workers)
    src_loaders = {
        d: load_pacs_test_data(args.datasetRoot, d, args.batch_size, args.num_workers)[0]
        for d in available
    }

    # peek args + num_class from node 0 checkpoint
    ck0 = torch.load(node_files[0], map_location="cpu", weights_only=False)
    ckpt_args = ck0["args"]
    num_class = ck0["backbone_state"]["backbone.fc.weight"].shape[0]
    print(f"num_class={num_class}")
    backbone = build_backbone(ckpt_args, num_class, device)

    A, own_acc = {}, {}
    Sxdom_acc, Sxdom_negent = {}, {}
    bn_early, bn_all = {}, {}  # node -> (mu, var)

    for j in range(args.num_nodes):
        load_backbone_only(node_files[j], backbone, device)
        backbone.eval()
        # A_j = target acc of node j (= broadcast BN_j equivalent)
        A[j] = eval_acc(backbone, target_loader)
        # BN vectors
        bn_early[j] = extract_bn_vectors(backbone, "early")
        bn_all[j] = extract_bn_vectors(backbone, "all")
        # cross-source: other source domains (not node j's own domain)
        own = node_to_domain[f"node_{j}"]
        own_acc[j] = eval_acc(backbone, src_loaders[own])
        others = [d for d in available if d != own]
        accs = [eval_acc(backbone, src_loaders[d]) for d in others]
        ents = [eval_mean_entropy(backbone, src_loaders[d], device) for d in others]
        Sxdom_acc[j] = float(np.mean(accs))
        Sxdom_negent[j] = -float(np.mean(ents))
        print(f"  node_{j}({own:>11}): A_j(target)={A[j]:6.2f}  own_src={own_acc[j]:6.2f}  "
              f"xdom_acc={Sxdom_acc[j]:6.2f}  xdom_negent={Sxdom_negent[j]:+.3f}")

    nodes = list(range(args.num_nodes))
    A_arr = np.array([A[j] for j in nodes])
    baseline_mean = float(A_arr.mean())
    oracle = float(A_arr.max())

    # all-avg candidate: average all BN buffers, eval target
    load_backbone_only(node_files[0], backbone, device)
    bufdict = dict(backbone.named_buffers())
    acc_buffers = {n: [] for n in bufdict if n.endswith(("running_mean", "running_var"))}
    for j in nodes:
        load_backbone_only(node_files[j], backbone, device)
        bd = dict(backbone.named_buffers())
        for n in acc_buffers:
            acc_buffers[n].append(bd[n].detach().clone())
    load_backbone_only(node_files[0], backbone, device)
    bd = dict(backbone.named_buffers())
    for n, lst in acc_buffers.items():
        bd[n].copy_(torch.stack(lst, 0).mean(0))
    backbone.eval()
    all_avg_acc = eval_acc(backbone, target_loader)

    # ---- R4 sanity: BN clustering (intra vs inter domain distance) ----
    mus = np.stack([bn_all[j][0] for j in nodes])
    vars_ = np.stack([bn_all[j][1] for j in nodes])
    Dall = dist_W2(mus, vars_)
    dom_of = [node_to_domain[f"node_{j}"] for j in nodes]
    intra = [Dall[i, k] for i in nodes for k in nodes if i < k and dom_of[i] == dom_of[k]]
    inter = [Dall[i, k] for i in nodes for k in nodes if i < k and dom_of[i] != dom_of[k]]
    intra_m, inter_m = float(np.mean(intra)), float(np.mean(inter))
    print(f"  [R4] BN W2 intra-domain={intra_m:.3f}  inter-domain={inter_m:.3f}  "
          f"ratio={inter_m / max(intra_m, 1e-9):.2f}  (expect inter>>intra=3 群)")

    # ---- assemble all signals (oriented so higher = predicted better) ----
    signals = {}
    for layerset, bn in (("early", bn_early), ("all", bn_all)):
        mus = np.stack([bn[j][0] for j in nodes])
        vars_ = np.stack([bn[j][1] for j in nodes])
        reps = np.concatenate([mus, vars_], axis=1)
        D_l2 = dist_L2_std(reps)
        D_w2 = dist_W2(mus, vars_)
        for metric, D in (("L2std", D_l2), ("W2", D_w2)):
            cen, prof = geometry_signals(D)
            signals[f"centroid_{metric}_{layerset}"] = cen
            signals[f"prof_{metric}_{layerset}"] = prof
        signals[f"varmag_{layerset}"] = np.array([float(np.mean(bn[j][1])) for j in nodes])
    signals["xdom_acc"] = np.array([Sxdom_acc[j] for j in nodes])
    signals["xdom_negent"] = np.array([Sxdom_negent[j] for j in nodes])

    # ---- per-node CSV rows ----
    for j in nodes:
        row = {"fold": target, "node": j, "domain": dom_of[j],
               "A_target": A[j], "own_src_acc": own_acc[j]}
        for name, vec in signals.items():
            row[name] = float(vec[j])
        out_rows.append(row)

    # ---- per-signal R1/R2 (Spearman + headroom recovery) ----
    print(f"  baseline_mean={baseline_mean:.2f}  all_avg={all_avg_acc:.2f}  oracle(max A_j)={oracle:.2f}")
    headroom = oracle - baseline_mean
    for name, vec in signals.items():
        rho = spearman(vec, A_arr)
        j_star = int(np.argmax(vec))
        recov = (A[j_star] - baseline_mean) / headroom if headroom > 1e-9 else float("nan")
        summary_rows.append({
            "fold": target, "signal": name, "spearman": rho,
            "argmax_node": j_star, "argmax_acc": A[j_star],
            "recovery_frac": recov, "baseline_mean": baseline_mean,
            "all_avg": all_avg_acc, "oracle": oracle,
            "intra_W2": intra_m, "inter_W2": inter_m,
        })
        print(f"    {name:24s}  rho={rho:+.3f}  argmax=node_{j_star}({dom_of[j_star]:>11}) "
              f"acc={A[j_star]:6.2f}  recov={recov:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasetRoot", default="../datasets/")
    ap.add_argument("--checkpoint_root", default=".")
    ap.add_argument("--folds", default="art_painting,cartoon,photo,sketch")
    ap.add_argument("--num_nodes", type=int, default=9)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    out_dir = args.output_dir or os.path.join("research", "outputs", ts, "bn_signal_gating")
    os.makedirs(out_dir, exist_ok=True)

    out_rows, summary_rows = [], []
    for target in args.folds.split(","):
        target = target.strip()
        run_fold(target, args, device, out_rows, summary_rows)

    # write CSVs
    per_node_csv = os.path.join(out_dir, "per_node_signals.csv")
    summary_csv = os.path.join(out_dir, "signal_rank_correlation.csv")
    if out_rows:
        with open(per_node_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader(); w.writerows(out_rows)
    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)

    # ---- cross-fold sign consistency (4-fold same-sign gate) ----
    print("\n=========== CROSS-FOLD SUMMARY (Spearman per signal) ===========")
    sig_names = sorted({r["signal"] for r in summary_rows})
    folds = [t.strip() for t in args.folds.split(",")]
    print(f"{'signal':24s} " + " ".join(f"{f[:6]:>7s}" for f in folds) + "  mean  same_sign  mean_recov")
    for name in sig_names:
        rhos = {r["fold"]: r["spearman"] for r in summary_rows if r["signal"] == name}
        recs = [r["recovery_frac"] for r in summary_rows if r["signal"] == name]
        vals = [rhos.get(f, float("nan")) for f in folds]
        same = all(v > 0 for v in vals) or all(v < 0 for v in vals)
        print(f"{name:24s} " + " ".join(f"{v:+7.3f}" for v in vals) +
              f"  {np.nanmean(vals):+.3f}  {'YES' if same else 'no':>8s}  {np.nanmean(recs):+.2f}")

    print(f"\nCSV → {per_node_csv}\n      {summary_csv}")
    print("判讀（plan §7）：強訊號 = Spearman>=0.6 且 4-fold same_sign=YES 且 mean_recov>=0.70 且 >all_avg。")


if __name__ == "__main__":
    main()
