"""階段 1 的原型幾何體檢（post-hoc、零訓練、零資料載入）。

問題（0811，1a 跑完後）：1a 的 proto_angle 四軸輸給 λ=0 的免費 energy，
在提下一臂之前必須先知道**原型本身長什麼樣**，而不是憑推測選 1b/2a。

⚠️ 與 `prototype_drift_probe.py` 的差別（**兩個不同的量、不可互相取代**）：
    drift probe  ：用節點自己的訓練子集**重新估**原型（post-hoc 推論，從未參與訓練）
    本 probe     ：讀 checkpoint 裡**真正參與過訓練**的 EMA buffer `prototypes`
  drift probe 回答「照搬 CIDER 會漂多少」；本 probe 回答「我們訓出來的原型現在是什麼形狀」。
  重用它的 angle_deg / relative_divergence（同一把尺，可與 BN-DIV 0.345 並排）。

量五項
------
1. 跨節點**同類別**原型夾角：跨畫風對 vs 同畫風對
   → 同畫風對＝取樣噪聲地板（三節點持不同子集）。階段 2 聚合的可行性直接看這兩者的差距。
2. 每節點 6 個類別中心兩兩夾角 → `L_disp` 實際推開到幾度（loss 只能反推出鬆界）
3. 原型空間 vs fc 空間的**類別幾何同構性**
   ⚠️ 有投影層時 128 vs 512 不可直接比夾角（train.py:1495 已擋）⇒ 改比兩個 6x6 夾角矩陣的
      Spearman 相關。高相關 ⇒ 投影層只是把分類器的類別幾何複製一份、沒提供額外資訊。
4. `proj_head.*` 跨節點相對分歧 → 驗證投影層確實有被 MH 聚合（計畫 §4.1 第 3 項）
   對照組：conv/fc（同為 trainable，應同級）、BN running（agg_bn 開時亦聚合）
5. `proto_count` 填充量 → EMA 樣本數是否足夠

⚠️ 原型是 buffer；communicator.py:976,1046 的 aggregate_bn 只取 running_mean/var 結尾，
   故原型**不會**被誤聚合（已查證）。階段 1 每節點只有自己畫風那一欄有值（6/18 格）。

用法：
  venv_matcha/bin/python scripts/proto_geometry_probe.py \
    --leave_out cartoon --checkpoint_dir exp_result_<desc> --description <desc> \
    --output_csv research/prototype_probe/0811_1a_proto_geometry.csv
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

from prototype_drift_probe import angle_deg, relative_divergence

PACS = ["art_painting", "cartoon", "photo", "sketch"]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v * np.nan


def spearman(a, b):
    """無 scipy 依賴的 Spearman：先轉秩再算 Pearson。"""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(np.dot(ra, rb) / d) if d > 1e-12 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leave_out", required=True, choices=PACS)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--num_nodes", type=int, default=9)
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--ckpt_epoch", default="final")
    p.add_argument("--output_csv", required=True)
    args = p.parse_args()

    suffix = "final" if str(args.ckpt_epoch) == "final" else f"epoch_{args.ckpt_epoch}"

    # node -> 來源域（與 fullspectrum_probe.py 同一段邏輯，已由訓練 log 逐項驗證）
    available = [d for d in PACS if d != args.leave_out]
    per = args.num_nodes // len(available)
    node_src = {i: available[min(i // per, len(available) - 1)] for i in range(args.num_nodes)}
    dom_idx = {d: i for i, d in enumerate(available)}   # 原型第二軸＝來源畫風

    protos, counts, states = {}, {}, {}
    for i in range(args.num_nodes):
        path = os.path.join(args.checkpoint_dir, f"{args.description}_node_{i}_{suffix}.pth")
        sd = torch.load(path, map_location="cpu", weights_only=False)["backbone_state"]
        states[i] = sd
        protos[i] = sd["prototypes"].float().numpy()      # [C, D, dim]
        counts[i] = sd["proto_count"].float().numpy()     # [C, D]

    rows = []
    C = args.num_classes

    # ---- 5. 填充量 -------------------------------------------------------------
    print("=== [5] proto_count 填充量 ===")
    for i in range(args.num_nodes):
        d = dom_idx[node_src[i]]
        cnt = counts[i][:, d]
        filled = int((counts[i] > 0).sum())
        print(f"  node_{i} ({node_src[i]:12}) cells={filled}/{C*len(available)} "
              f"該欄每類樣本數 min={cnt.min():.0f} mean={cnt.mean():.1f} max={cnt.max():.0f}")
        rows.append(dict(metric="proto_count", node=f"node_{i}", own_src=node_src[i],
                         value=float(cnt.mean()), detail=f"min={cnt.min():.0f},max={cnt.max():.0f},cells={filled}"))

    # 每個節點的「類別中心」＝它自己畫風那一欄（階段 1 只有一欄有值）
    centers = {}
    for i in range(args.num_nodes):
        d = dom_idx[node_src[i]]
        centers[i] = np.stack([unit(protos[i][c, d]) for c in range(C)])   # [C, dim]

    # ---- 1. 跨節點同類別原型夾角：跨畫風 vs 同畫風 -------------------------------
    print("\n=== [1] 跨節點同類別原型夾角（階段 2 聚合可行性）===")
    cross, within = [], []
    for i, j in itertools.combinations(range(args.num_nodes), 2):
        angs = [angle_deg(centers[i][c], centers[j][c]) for c in range(C)]
        m = float(np.nanmean(angs))
        same = node_src[i] == node_src[j]
        (within if same else cross).append(m)
        rows.append(dict(metric="pair_same_class_angle_deg", node=f"node_{i}-node_{j}",
                         own_src=f"{node_src[i]}|{node_src[j]}", value=m,
                         detail="within_domain" if same else "cross_domain"))
    print(f"  同畫風對（取樣噪聲地板） n={len(within):2d}  mean={np.mean(within):6.2f}° "
          f"[{np.min(within):.2f}, {np.max(within):.2f}]")
    print(f"  跨畫風對（畫風+噪聲）   n={len(cross):2d}  mean={np.mean(cross):6.2f}° "
          f"[{np.min(cross):.2f}, {np.max(cross):.2f}]")
    print(f"  ⇒ 畫風造成的超額夾角 = {np.mean(cross) - np.mean(within):.2f}°")

    # ---- 2. 每節點類別中心兩兩夾角（L_disp 的實際成果）---------------------------
    print("\n=== [2] 每節點 6 個類別中心兩兩夾角（L_disp 實際推開程度）===")
    print(f"  理論上限：6 類單純形 cos=-0.2 ⇒ {np.degrees(np.arccos(-0.2)):.2f}°")
    for i in range(args.num_nodes):
        angs = [angle_deg(centers[i][a], centers[i][b]) for a, b in itertools.combinations(range(C), 2)]
        print(f"  node_{i} ({node_src[i]:12}) min={np.min(angs):6.2f}° mean={np.mean(angs):6.2f}° max={np.max(angs):6.2f}°")
        rows.append(dict(metric="inter_class_center_angle_deg", node=f"node_{i}", own_src=node_src[i],
                         value=float(np.mean(angs)), detail=f"min={np.min(angs):.2f},max={np.max(angs):.2f}"))

    # ---- 3. 原型空間 vs fc 空間的類別幾何同構性 ---------------------------------
    print("\n=== [3] 原型空間 vs fc 空間的類別幾何（6x6 夾角矩陣 Spearman）===")
    print("  高相關 ⇒ 投影層只複製了分類器的類別幾何、沒提供額外資訊")
    for i in range(args.num_nodes):
        w = states[i]["backbone.fc.weight"].float().numpy()[:C]
        wn = np.stack([unit(w[c]) for c in range(C)])
        pa, fa = [], []
        for a, b in itertools.combinations(range(C), 2):
            pa.append(angle_deg(centers[i][a], centers[i][b]))
            fa.append(angle_deg(wn[a], wn[b]))
        r = spearman(np.array(pa), np.array(fa))
        print(f"  node_{i} ({node_src[i]:12}) spearman={r:+.3f}   fc 類間夾角 mean={np.mean(fa):.2f}°")
        rows.append(dict(metric="proto_vs_fc_geometry_spearman", node=f"node_{i}", own_src=node_src[i],
                         value=r, detail=f"fc_inter_class_mean={np.mean(fa):.2f}"))

    # ---- 4. proj_head 跨節點分歧（有沒有真的被聚合）------------------------------
    print("\n=== [4] 跨節點參數分歧 relative_divergence（同 BN-DIV 尺規）===")
    groups = {
        "proj_head": [k for k in states[0] if k.startswith("proj_head") and k.endswith("weight")],
        "fc":        ["backbone.fc.weight"],
        "conv1":     ["backbone.conv1.weight"],
        "layer4":    [k for k in states[0] if k.startswith("backbone.layer4") and k.endswith("weight") and states[0][k].dim() == 4],
        "bn_running": [k for k in states[0] if k.endswith("running_mean")],
    }
    for name, keys in groups.items():
        if not keys:
            continue
        divs = []
        for k in keys:
            vecs = [states[i][k].float().numpy().ravel() for i in range(args.num_nodes)]
            divs.append(relative_divergence(vecs))
        v = float(np.nanmean(divs))
        print(f"  {name:11} n_tensor={len(keys):3d}  rel_div={v:.3e}")
        rows.append(dict(metric="param_rel_divergence", node="ALL", own_src=name,
                         value=v, detail=f"n_tensor={len(keys)}"))

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    new = not os.path.exists(args.output_csv)
    with open(args.output_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "metric", "node", "own_src", "value", "detail"])
        if new:
            w.writeheader()
        for r in rows:
            r["run"] = args.description
            w.writerow(r)
    print(f"\nAppended {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
