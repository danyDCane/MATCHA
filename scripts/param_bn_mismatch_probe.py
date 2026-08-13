"""參數-BN 失配 介入探針（post-hoc、零重訓）。

命題：sync ring 泛化差(69.63) 是「共識(被平均)參數 × local BN」失配、非參數品質。
設計（載 sync ring / async ring 各 9 節點 final checkpoint）：
  A 原生      = 節點 j 參數 × 節點 j local BN           （= 現有 per-node target acc）
  B 共識×local = 9 節點參數平均 × 節點 j local BN         （人工製造失配態）
判定：
  - async-B ≪ async-A                 → 平均參數配 local BN 會掉 = 失配傷泛化
  - async-B ≈ sync-A(原生 69.63)      → sync 差就是因為它終態已是「共識參數×local BN」
  - 反之 B≈A                          → 失配否證、另尋機制

機制對應：sync 聚合 model.parameters()(含 BN affine weight/bias)、不動 BN running buffer。
故「B = 平均 named_parameters、保留 node j 的 running_mean/var buffer」精確對應 sync 終態。

用法：
  venv_matcha/bin/python scripts/param_bn_mismatch_probe.py \
      --datasetRoot ../datasets/ --target cartoon
"""
import argparse
import glob
import os
import numpy as np
import torch

from bn_signal_gating import (
    build_backbone, load_backbone_only, eval_acc, reconstruct_node_to_domain,
)
from test_domain_ood_scores import load_pacs_test_data

# (label, checkpoint dir) —— ring 兩設定
SETTINGS = [
    ("sync_ring", "exp_result_v1_stage2_leave_cartoon_ring"),
    ("async_ring", "exp_result_v1_stage2_leave_cartoon_async_const_tau1e-5_style_ring"),
]


def node_files(ckpt_dir, num_nodes):
    files = {}
    for j in range(num_nodes):
        hits = glob.glob(os.path.join(ckpt_dir, f"*node_{j}_final*.pth"))
        if not hits:
            raise FileNotFoundError(f"missing node {j} final ckpt in {ckpt_dir}")
        files[j] = sorted(hits)[0]
    return files


@torch.no_grad()
def compute_avg_params(backbone, files, nodes, device):
    """9 節點 named_parameters 逐鍵平均（含 BN affine、排除 running buffer）。"""
    stacks = {}
    for j in nodes:
        load_backbone_only(files[j], backbone, device)
        for name, p in backbone.named_parameters():
            stacks.setdefault(name, []).append(p.data.detach().clone())
    return {name: torch.stack(lst, 0).mean(0) for name, lst in stacks.items()}


@torch.no_grad()
def eval_combo(backbone, node_file, override_params, target_loader, device):
    """載 node 的 backbone(=拿它的 local BN buffer)；可選用 override_params 覆蓋 named_parameters。"""
    load_backbone_only(node_file, backbone, device)
    if override_params is not None:
        pd = dict(backbone.named_parameters())
        for name, val in override_params.items():
            pd[name].data.copy_(val)
    backbone.eval()
    return eval_acc(backbone, target_loader)


def run_setting(label, ckpt_dir, args, device, target_loader):
    files = node_files(ckpt_dir, args.num_nodes)
    nodes = list(range(args.num_nodes))
    ck0 = torch.load(files[0], map_location="cpu", weights_only=False)
    num_class = ck0["backbone_state"]["backbone.fc.weight"].shape[0]
    backbone = build_backbone(ck0["args"], num_class, device)

    avg_params = compute_avg_params(backbone, files, nodes, device)

    A, B = {}, {}
    for j in nodes:
        A[j] = eval_combo(backbone, files[j], None, target_loader, device)          # 原生
        B[j] = eval_combo(backbone, files[j], avg_params, target_loader, device)    # 共識參數×local BN
        print(f"  [{label}] node_{j}: A(orig)={A[j]:6.2f}  B(avg-param×localBN)={B[j]:6.2f}  ΔB-A={B[j]-A[j]:+6.2f}")

    A_arr = np.array([A[j] for j in nodes])
    B_arr = np.array([B[j] for j in nodes])
    print(f"  [{label}] A_mean={A_arr.mean():6.2f}  B_mean={B_arr.mean():6.2f}  Δ(B-A)={B_arr.mean()-A_arr.mean():+6.2f}")
    return {"A_mean": float(A_arr.mean()), "B_mean": float(B_arr.mean()),
            "A": A_arr.tolist(), "B": B_arr.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_root", default=".")
    ap.add_argument("--datasetRoot", default="../datasets/")
    ap.add_argument("--target", default="cartoon")
    ap.add_argument("--num_nodes", type=int, default=9)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reconstruct_node_to_domain(args.target, args.num_nodes)  # sanity (raises if mismatch)
    target_loader, _ = load_pacs_test_data(args.datasetRoot, args.target, args.batch_size, args.num_workers)

    res = {}
    for label, sub in SETTINGS:
        ckpt_dir = os.path.join(args.checkpoint_root, sub)
        print(f"\n===== {label}  ({sub}) =====")
        res[label] = run_setting(label, ckpt_dir, args, device, target_loader)

    print("\n================ 判定 ================")
    sA, sB = res["sync_ring"]["A_mean"], res["sync_ring"]["B_mean"]
    aA, aB = res["async_ring"]["A_mean"], res["async_ring"]["B_mean"]
    print(f"sync : A(orig)={sA:.2f}  B(avg×localBN)={sB:.2f}  Δ={sB-sA:+.2f}")
    print(f"async: A(orig)={aA:.2f}  B(avg×localBN)={aB:.2f}  Δ={aB-aA:+.2f}")
    print(f"→ 核心檢驗 async-B({aB:.2f}) vs sync-A原生({sA:.2f})：差 {aB-sA:+.2f}")
    print("  若 async-B ≪ async-A 且 async-B ≈ sync-A → 參數-BN 失配成立")


if __name__ == "__main__":
    main()
