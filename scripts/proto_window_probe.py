"""原型抖動的來源拆解：是「EMA 窗口太短」還是「各節點資料本來就不同」？

問題（2026-08-13 dany）：階段 2 的訊噪比只有 1:1——同畫風節點對的原型差距 **10.23°**
（純抖動）與畫風造成的超額 **10.89°** 幾乎一樣大。若要做階段 2b，得先把抖動壓下來。
但「提高 proto_m 就能壓」這個假設**從未驗證**：10.23° 有兩個來源混在一起——

  (A) EMA 窗口太短：proto_m=0.95 ⇒ 原型只由少量最近樣本決定
  (B) 各節點持有不同的訓練子集：三個同畫風節點拿到的是同一個域的不同切片

若主因是 (A) ⇒ 提高 proto_m 有效；若是 (B) ⇒ 提高 proto_m 沒用（資料本來就不同）。

作法
----
用 post-hoc 特徵**模擬不同的 EMA 窗口**。指數加權平均的變異數等效於均勻平均 k 個樣本：

    Var(EMA_m) = (1-m)/(1+m) · Var(z)  =  Var(z)/k     ⇒   k = (1+m)/(1-m)

    proto_m  0.90 → k=19 ｜ **0.95（現用）→ k=39** ｜ 0.99 → k=199 ｜ 0.995 → k=399

⇒ 掃 k、看同畫風節點對的夾角怎麼降：
    k 增大時夾角**大幅下降** ⇒ 主因是 (A)，提高 proto_m 有效
    k 增大到「用完全部樣本」仍居高不下 ⇒ 主因是 (B)，提高 proto_m 無效

★ 所有 k 都在**同一次前向、同一個 BN 模式、同一個投影空間**下算 ⇒ **組內零 confound**。
  這是本腳本的核心設計：不跟 checkpoint 裡的 EMA 原型跨條件比，而是自己造對照。

⚠️ 與 checkpoint EMA 原型（10.23°）的對照只能當**參考**，不可當結論依據：
   訓練時的原型是 train-mode BN（batch 統計）算的，本腳本是 eval-mode BN（running 統計）
   ⇒ 兩者落在不同的正規化座標系（0805 探針抓到的同型問題）。**主結論一律取自組內 k 的趨勢。**

⚠️ 必須用**投影後**的 128 維空間（`model.project()`），不是 512 維 penultimate——
   checkpoint 裡的原型活在投影空間，用 512 維算出來的是另一個量。
   （`prototype_drift_probe.py` 用的是 512 維，因為它寫於投影層存在之前。）

重用：`prototype_drift_probe.py` 的 `rebuild_node_subsets`（精確重現訓練切分）與 `angle_deg`。

用法：
  venv_matcha/bin/python scripts/proto_window_probe.py \
    --leave_out cartoon --checkpoint_dir exp_result_<desc> --description <desc> \
    --output_csv research/prototype_probe/0813_1a_proto_window.csv
"""
import os
import sys
import csv
import argparse
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import util
from prototype_drift_probe import rebuild_node_subsets, angle_deg
from osdg_eval import load_backbone_diffusion

PACS = ["art_painting", "cartoon", "photo", "sketch"]


@torch.no_grad()
def extract_features(backbone, subset, device, batch_size, num_workers):
    """抽出投影後的 128 維特徵（與 checkpoint 裡的原型同空間）與標籤。"""
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    Z, Y = [], []
    for batch in loader:
        data, y, _ = util.unpack_batch(batch)
        data = data.to(device, non_blocking=True)
        z3 = backbone.forward_to_layer3_style(data, communicator=None)
        _, vec = backbone.forward_from_layer3(z3)
        Z.append(backbone.project(vec).cpu().numpy())     # ★ 投影後、已 L2 正規化
        Y.append(np.asarray(y).flatten())
    return np.concatenate(Z), np.concatenate(Y)


def protos_from_subsample(Z, Y, C, k, rng):
    """每類隨機抽 k 個樣本算原型（k<=0 或不足則用該類全部）。回傳 [C, P]，缺類為 NaN。"""
    P = Z.shape[1]
    out = np.full((C, P), np.nan)
    for c in range(C):
        idx = np.flatnonzero(Y == c)
        if len(idx) == 0:
            continue
        take = idx if (k <= 0 or k >= len(idx)) else rng.choice(idx, k, replace=False)
        v = Z[take].mean(axis=0)
        n = np.linalg.norm(v)
        if n > 1e-12:
            out[c] = v / n
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leave_out", required=True, choices=PACS)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--datasetRoot", default="../datasets/")
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--exclude_class_idx", type=int, default=6)
    p.add_argument("--num_nodes", type=int, default=9)
    p.add_argument("--randomSeed", type=int, default=2026)
    p.add_argument("--node_split_mode", default="class_balanced")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt_epoch", default="final")
    p.add_argument("--repeats", type=int, default=20, help="每個 k 的重抽樣次數")
    p.add_argument("--ks", default="19,39,79,199,399,0",
                   help="等效樣本數清單；0=用該類全部樣本（無窗口上限）")
    p.add_argument("--probe_seed", type=int, default=0)
    p.add_argument("--output_csv", required=True)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    suffix = "final" if str(args.ckpt_epoch) == "final" else f"epoch_{args.ckpt_epoch}"
    C = args.num_classes
    ks = [int(x) for x in args.ks.split(",")]
    m_of_k = {19: 0.90, 39: 0.95, 79: 0.975, 199: 0.99, 399: 0.995}

    node_to_domain, _, node_subsets = rebuild_node_subsets(args)

    # ---- 一次前向，抽出每個節點的全部特徵（之後所有 k 都在這批特徵上重抽樣）----
    feats = {}
    for i in range(args.num_nodes):
        node = f"node_{i}"
        ckpt = os.path.join(args.checkpoint_dir, f"{args.description}_{node}_{suffix}.pth")
        backbone, _ = load_backbone_diffusion(ckpt, C, device)
        if not hasattr(backbone, "prototypes"):
            raise RuntimeError(f"{ckpt} 沒有 prototypes buffer——該 run 不是用 --use_proto_reg 訓練的。")
        Z, Y = extract_features(backbone, node_subsets[node], device, args.batch_size, args.num_workers)
        feats[node] = (Z, Y)
        print(f"[feat] {node} ({node_to_domain[node]:12}) n={len(Y)} dim={Z.shape[1]} "
              f"每類樣本數={[int((Y == c).sum()) for c in range(C)]}")
        del backbone

    pairs = list(itertools.combinations(range(args.num_nodes), 2))
    rows = []
    print(f"\n{'k(等效樣本)':>12}{'proto_m':>9}{'同畫風對':>10}{'跨畫風對':>10}{'畫風超額':>10}   (度)")
    for k in ks:
        rng = np.random.default_rng(args.probe_seed)
        within_r, cross_r = [], []
        percls = {c: [] for c in range(C)}
        for _ in range(args.repeats):
            P = {f"node_{i}": protos_from_subsample(*feats[f"node_{i}"], C, k, rng)
                 for i in range(args.num_nodes)}
            for i, j in pairs:
                a, b = P[f"node_{i}"], P[f"node_{j}"]
                angs = [angle_deg(a[c], b[c]) for c in range(C)]
                same = node_to_domain[f"node_{i}"] == node_to_domain[f"node_{j}"]
                (within_r if same else cross_r).append(float(np.nanmean(angs)))
                if same:
                    for c in range(C):
                        if not np.isnan(angs[c]):
                            percls[c].append(angs[c])
        w, x = float(np.mean(within_r)), float(np.mean(cross_r))
        tag = f"{k}" if k > 0 else "全部"
        mtag = f"{m_of_k[k]:.3f}" if k in m_of_k else ("—" if k <= 0 else "—")
        print(f"{tag:>12}{mtag:>9}{w:>10.2f}{x:>10.2f}{x - w:>10.2f}")
        rows.append(dict(run=args.description, k=tag, proto_m_equiv=mtag,
                         within_domain_deg=round(w, 4), cross_domain_deg=round(x, 4),
                         style_excess_deg=round(x - w, 4), repeats=args.repeats,
                         **{f"within_cls{c}_deg": round(float(np.mean(percls[c])), 4)
                            for c in range(C) if percls[c]}))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    new = not os.path.exists(args.output_csv)
    with open(args.output_csv, "a", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nAppended {len(rows)} rows to {args.output_csv}")
    print("\n[逐類別・同畫風對的抖動]（看稀有類別是否受限於樣本數不足）")
    hdr = "".join(f"cls{c}".rjust(9) for c in range(C))
    print(f"{'k':>12}{hdr}")
    for r in rows:
        print(f"{r['k']:>12}" + "".join(f"{r.get(f'within_cls{c}_deg', float('nan')):9.2f}" for c in range(C)))


if __name__ == "__main__":
    main()
