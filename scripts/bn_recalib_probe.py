"""BN recalibration 歸因探針（post-hoc、零重訓、不覆寫 checkpoint）。

問題：sketch 節點 sync=54 vs async=60 的泛化差，是「BN 域錯配」還是「權重品質」？
設計（每 setting 每節點、target=cartoon）：
  orig     = 節點原生 BN（訓練累積的 sketch 統計）→ 必須重現 acc.log 末 epoch（sanity）
  oracleBN = 凍權重、BN **重估於 cartoon(target)** → 「若 BN 對 target 是對的、權重本身多好」
判定（比 sketch group r6-8，sync/model_only/full_async 三方在**相同 oracle-BN 待遇**下）：
  oracleBN 後三方拉平且相等        → 差是 BN 域錯配（BN 是主因、dany 對）
  oracleBN 後 async 權重仍 > sync  → 差在權重本身、BN 不是全部
附帶（probe 2）：bn_shift = ‖orig−oracle‖ = 該節點 BN 離「cartoon-正確」多遠；
  async sketch 的 shift 若 < sync → async 訓出來的 BN 本來就更可遷移。

⚠ 關鍵正確性：MATCHA resnet 在 self.training=True 會觸發隨機 style_shift，會污染 BN 重估。
  故重估時 model.eval()（關 style_shift）、只把 BN 子模組設 train 更新、momentum=None 累積平均。
⚠ 用 target 測試集同時校 BN 與評估 = transductive/test-time BN；對「歸因」而言三方待遇相同故比較公平，
  絕對值會被 test-time 抬高，看的是 **sync-vs-async 在相同待遇下是否拉平**。

用法：
  venv_matcha/bin/python scripts/bn_recalib_probe.py --datasetRoot ../datasets/
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

import util
from bn_signal_gating import (
    build_backbone, load_backbone_only, eval_acc,
    reconstruct_node_to_domain, extract_bn_vectors,
)
from test_domain_ood_scores import load_pacs_test_data

# (label, checkpoint dir) —— 全 seed2026 + topo1234、cartoon leave-out
SETTINGS = [
    ("sync",       "exp_result_v1_stage2_leave_cartoon_seed2026_topo1234"),
    ("model_only", "exp_result_v1_stage2_leave_cartoon_async_const_tau1e-5_seed2026_topo1234_modelonly"),
    ("full_async", "exp_result_v1_stage2_leave_cartoon_async_const_tau1e-5_style_seed2026_topo1234"),
]
SKETCH_NODES = [6, 7, 8]


def bn_modules(model):
    return [m for m in model.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]


def capture_bn(model):
    mu, var = extract_bn_vectors(model, "all")
    return np.concatenate([mu, var])


@torch.no_grad()
def recompute_bn(model, loader, device, max_batches=None):
    """凍權重、把 BN running stats 重估於 loader 的資料。
    model.eval() 確保 style_shift 關閉；只將 BN 子模組設 train 以更新 running stats。"""
    model.eval()                       # style_shift OFF（ResNet.self.training=False）
    bns = bn_modules(model)
    saved_mom = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()        # running_mean=0, running_var=1, num_batches_tracked=0
        m.momentum = None              # 累積移動平均（與 batch 順序無關、確定性）
        m.train()                      # 這些 BN 在 forward 時更新（父模組仍 eval → style_shift 不觸發）
    nb = 0
    for batch in loader:
        inputs, _, _ = util.unpack_batch(batch)
        inputs = inputs.to(device, non_blocking=True)
        model(inputs)                  # 只 forward、更新 BN
        nb += 1
        if max_batches is not None and nb >= max_batches:
            break
    for m, mom in zip(bns, saved_mom):
        m.momentum = mom
        m.eval()                       # 復原：BN 回 eval、之後評估用重估後的 running stats
    return nb


def find_res_dir(ckpt_dir):
    hits = glob.glob(ckpt_dir + "v*_res")
    return hits[0] if hits else None


def node_files(ckpt_dir, num_nodes):
    files = {}
    for j in range(num_nodes):
        hits = glob.glob(os.path.join(ckpt_dir, f"*node_{j}_final*.pth"))
        if not hits:
            raise FileNotFoundError(f"missing node {j} final ckpt in {ckpt_dir}")
        files[j] = sorted(hits)[0]
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasetRoot", default="../datasets/")
    ap.add_argument("--target", default="cartoon")
    ap.add_argument("--num_nodes", type=int, default=9)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_to_domain, available = reconstruct_node_to_domain(args.target, args.num_nodes)
    print(f"target(leave-out)={args.target}  available={available}")
    print(f"node->domain={node_to_domain}")
    target_loader, _ = load_pacs_test_data(args.datasetRoot, args.target,
                                           args.batch_size, args.num_workers)

    results = {}
    sanity_bad = 0
    determinism_checked = False

    for label, ckpt_dir in SETTINGS:
        res_dir = find_res_dir(ckpt_dir)
        files = node_files(ckpt_dir, args.num_nodes)
        ck0 = torch.load(files[0], map_location="cpu", weights_only=False)
        num_class = ck0["backbone_state"]["backbone.fc.weight"].shape[0]
        backbone = build_backbone(ck0["args"], num_class, device)

        print(f"\n===== {label}  ({ckpt_dir}) =====")
        print(f"{'node':14s} {'acclog':>7s} {'orig':>7s} {'match':>9s} {'oracleBN':>9s} {'bn_shift':>9s}")
        rows = []
        for j in range(args.num_nodes):
            dom = node_to_domain[f"node_{j}"]
            acclog = float(np.loadtxt(os.path.join(
                res_dir, f"dsgd-lr0.001-budget1.0-r{j}-acc.log"))[-1]) if res_dir else float("nan")

            # orig-BN
            load_backbone_only(files[j], backbone, device)
            acc_orig = eval_acc(backbone, target_loader)
            bn_before = capture_bn(backbone)

            # oracle-BN（重估於 cartoon）
            load_backbone_only(files[j], backbone, device)  # 從乾淨 orig BN 重新開始
            recompute_bn(backbone, target_loader, device)
            bn_after = capture_bn(backbone)
            acc_oracle = eval_acc(backbone, target_loader)
            bn_shift = float(np.linalg.norm(bn_before - bn_after))

            # sanity 1: orig 對得上 acc.log 末 epoch?
            match = "OK" if (not np.isnan(acclog) and abs(acc_orig - acclog) < 1.5) else "⚠BAD"
            if match != "OK":
                sanity_bad += 1
            # sanity 2: BN 真的變了?
            if bn_shift <= 1e-6:
                match += "/BNnochg⚠"
            print(f"node_{j}({dom:>11}) {acclog:7.2f} {acc_orig:7.2f} {match:>9s} "
                  f"{acc_oracle:9.2f} {bn_shift:9.3f}")
            rows.append({"node": j, "dom": dom, "acclog": acclog,
                         "orig": acc_orig, "oracle": acc_oracle, "bn_shift": bn_shift})

            # sanity 3: 決定性（第一個 setting 的 node_8 重估兩次應一致）
            if not determinism_checked and label == SETTINGS[0][0] and j == 8:
                load_backbone_only(files[j], backbone, device)
                recompute_bn(backbone, target_loader, device)
                acc_oracle2 = eval_acc(backbone, target_loader)
                print(f"   [determinism] oracleBN run1={acc_oracle:.4f} run2={acc_oracle2:.4f} "
                      f"→ {'OK 確定性' if abs(acc_oracle-acc_oracle2)<1e-6 else '⚠非確定(style_shift漏關?)'}")
                determinism_checked = True

        results[label] = rows

    # ---- group summary（三群、重點 sketch）----
    def gmean(rows, nodes, key):
        return float(np.mean([r[key] for r in rows if r["node"] in nodes]))

    groups = {"art(0-2)": [0, 1, 2], "photo(3-5)": [3, 4, 5], "sketch(6-8)": SKETCH_NODES}
    print("\n================ 群組摘要（orig → oracleBN）================")
    print(f"{'group':12s} " + " ".join(f"{lab:>22s}" for lab, _ in SETTINGS))
    for gname, nodes in groups.items():
        cells = []
        for lab, _ in SETTINGS:
            o = gmean(results[lab], nodes, "orig")
            k = gmean(results[lab], nodes, "oracle")
            cells.append(f"{o:5.1f}→{k:5.1f}(Δ{k-o:+4.1f})")
        print(f"{gname:12s} " + " ".join(f"{c:>22s}" for c in cells))

    print("\n================ 判決（sketch group）================")
    sk = {lab: (gmean(results[lab], SKETCH_NODES, "orig"),
                gmean(results[lab], SKETCH_NODES, "oracle"),
                gmean(results[lab], SKETCH_NODES, "bn_shift")) for lab, _ in SETTINGS}
    for lab in ("sync", "model_only", "full_async"):
        o, k, s = sk[lab]
        print(f"  {lab:12s}: orig={o:5.2f}  oracleBN={k:5.2f}  (Δ{k-o:+.2f})  bn_shift={s:.3f}")
    o_gap = sk["full_async"][1] - sk["sync"][1]
    print(f"\n  ★ oracle-BN 後 full_async − sync = {o_gap:+.2f}")
    print(f"    |Δ|<~1 且三方接近 → 差是 BN 域錯配（BN 主因）；仍 >~2 → 差在權重本身")
    print(f"  ★ bn_shift(離cartoon正確): sync={sk['sync'][2]:.3f} vs async={sk['full_async'][2]:.3f}"
          f" → async 較小代表其 BN 本來就較可遷移")

    print(f"\n[sanity] orig-vs-acclog 不符節點數 = {sanity_bad}（>0 代表 pipeline 有問題、結果不可信）")


if __name__ == "__main__":
    main()
