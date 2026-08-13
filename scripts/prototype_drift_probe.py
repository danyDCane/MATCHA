"""階段 0：類別原型漂移的失效證明（post-hoc、零訓練）。

問題（0803 §4.0）：若照搬 CIDER/PALM 而**不做任何調整**，各節點的類別原型會漂到什麼程度？
這個漂移大到值得處理嗎？

背景：CIDER 的類原型 μ_c 是 **EMA buffer、不在 named_parameters()** ⇒ MATCHA 的 MH 聚合
不會碰它（communicator.py:713 註解原文：Only aggregates trainable parameters, NOT buffers）
⇒ 9 個節點的「類別中心」各自朝自身 source domain 漂移。更糟：compactness 損失會主動把
樣本拉向本地原型 ＝ 鼓勵每個節點學域專屬表徵，與 DG 目標相反。

以上是文獻推論（CIDER JSON 的 p2p_failure_mode），本腳本去拿自己的數字。

量什麼
------
對每個節點，用**它自己真實持有的訓練子集**估出該節點的類別原型（6 個，因為節點只持有一個域），
然後量節點兩兩之間、同一類別的原型夾角：

  跨域對 (cross-domain)：不同 source domain 的節點  → 主訊號（域差異 + 取樣噪聲）
  同域對 (within-domain)：相同 source domain 的節點  → **取樣噪聲地板**（對照組）

同域三節點持有的是**不同子集**（util.py:833 用 Subset(full, node_indices[node]))，
故同域夾角＝純取樣差異，是免費且必要的對照。

判準（0803 §4.0 事前寫死，事後不得回調）
------
  跨域 >= 3x 同域地板  → 失效證明成立，進階段 1
  跨域 ~= 同域         → 原型不聚合不構成問題，調整一失去必要性、原型路線須重評

原型定義（對齊 CIDER Eq.8 / PALM Eq.7）
------
特徵先 L2 正規化 → 按類別平均 → 再 L2 正規化。
（EMA 的極限即為平均，故此為原型的無偏估計。）

⚠️ 界線：本腳本量的是「終態 checkpoint 上**重估**的原型」，不是「訓練過程中 EMA 演化的原型」。
真實訓練中 compactness 損失會主動塑形，兩者不同。故本腳本只能證明「特徵空間中的類別中心
確實按域分開」，**不能**證明「訓練中的原型會漂多少」——後者要等階段 1。

純 post-hoc inference、零訓練、不改任何 checkpoint。
重用：osdg_eval.load_backbone_diffusion / util.assign_nodes_to_domains /
      util.partition_domain_dataset_for_nodes / pacs_dataset.PACSDataset /
      models/resnet.py:361 intermediate_forward（512 維 penultimate，即現行餵給 diffusion 的 latent）

用法：
  venv_matcha/bin/python scripts/prototype_drift_probe.py \
    --leave_out cartoon \
    --checkpoint_dir exp_result_v1_stage2_leave_cartoon_async_const_tau1e-5_style_osdg_excl_person_seed2026_topo1234 \
    --description v1_stage2_leave_cartoon_async_const_tau1e-5_style_osdg_excl_person_seed2026_topo1234 \
    --output_csv research/bn_fusion/0803_prototype_drift.csv
"""
import os
import sys
import csv
import argparse
import itertools
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

import util
from pacs_dataset import PACSDataset
from osdg_eval import load_backbone_diffusion

PACS = ["art_painting", "cartoon", "photo", "sketch"]

# 域距離（0724 報告，以 cartoon 為 target 時）。⚠️ 其度量定義尚未回查原文核實，
# 僅用於呈現排序、不作定量主張（同 0730 報告 §7 界線 3）。
DOMAIN_DIST_TO_CARTOON = {"art_painting": 22, "photo": 25, "sketch": 29}


def build_test_transform():
    """與 test_domain_ood_scores.load_pacs_test_data 一致的確定性 transform。
    ⚠️ 刻意不用訓練時的隨機增強：階段 0 要的是可重現的特徵估計。"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def rebuild_node_subsets(args):
    """精確重現訓練時每個節點持有的資料子集。

    照 util.py:780-836 的 PACS virtual-node 路徑逐步重現：
      1. assign_nodes_to_domains（contiguous：node 0-2 = 第一域, 3-5 = 第二域, ...）
      2. partition_domain_dataset_for_nodes(split_mode, seed=randomSeed)
      3. **再**過濾掉 exclude_class —— 順序不可顛倒（partition 在含 person 的完整集上做）
    """
    available = [d for d in PACS if d != args.leave_out]
    node_to_domain, domain_to_nodes = util.assign_nodes_to_domains(available, args.num_nodes)
    print(f"[setup] Node assignment: {node_to_domain}")

    tf = build_test_transform()
    full_datasets, node_subsets = {}, {}

    for domain in available:
        full_datasets[domain] = PACSDataset(
            root=args.datasetRoot, dataset_name=domain, transform=tf)

        node_names = domain_to_nodes[domain]
        node_indices = util.partition_domain_dataset_for_nodes(
            full_datasets[domain], node_names,
            split_mode=args.node_split_mode, seed=args.randomSeed)

        for node_name in node_names:
            idxs = node_indices[node_name]
            n_before = len(idxs)
            if args.exclude_class_idx is not None:
                targets = full_datasets[domain].targets
                idxs = [i for i in idxs if targets[i] != args.exclude_class_idx]
            node_subsets[node_name] = Subset(full_datasets[domain], idxs)
            print(f"[setup] {node_name} <- {domain}: {len(idxs)}/{n_before} samples "
                  f"(excluded class idx {args.exclude_class_idx})")

    return node_to_domain, domain_to_nodes, node_subsets


@torch.no_grad()
def compute_prototypes(backbone, subset, num_classes, device, batch_size, num_workers):
    """回傳 (protos[C, D] 單位向量或 NaN, counts[C])。

    對齊 CIDER Eq.8 / PALM Eq.7：特徵先 L2 正規化 → 按類別平均 → 再 L2 正規化。
    """
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    acc, counts, feat_dim = None, np.zeros(num_classes, dtype=np.int64), None

    for batch in loader:
        data, y, _ = util.unpack_batch(batch)
        data = data.to(device, non_blocking=True)
        feats = backbone.intermediate_forward(data)          # [B, 512] penultimate
        feats = torch.nn.functional.normalize(feats, dim=1)  # 單位球
        y = np.asarray(y).flatten()

        if acc is None:
            feat_dim = feats.shape[1]
            acc = torch.zeros(num_classes, feat_dim, dtype=torch.float64, device=device)
        for c in range(num_classes):
            m = (y == c)
            if m.any():
                acc[c] += feats[torch.from_numpy(m).to(device)].sum(0).double()
                counts[c] += int(m.sum())

    protos = torch.full((num_classes, feat_dim), float("nan"), dtype=torch.float64)
    for c in range(num_classes):
        if counts[c] > 0:
            v = acc[c] / counts[c]
            protos[c] = (v / v.norm()).cpu()
    return protos.numpy(), counts


def angle_deg(p, q):
    """兩個單位向量的夾角（度）。任一為 NaN 回 NaN。"""
    if np.isnan(p).any() or np.isnan(q).any():
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(np.dot(p, q)), -1.0, 1.0))))


def relative_divergence(vectors):
    """對齊 0628/BN-DIV 的定義 mean||v - mu|| / ||mu||，mu = 未正規化的平均。
    使其與 BN-DIV 0.345 可放在同一把尺上看。"""
    V = np.stack(vectors)
    if np.isnan(V).any():
        return float("nan")
    mu = V.mean(0)
    nmu = np.linalg.norm(mu)
    if nmu < 1e-12:
        return float("nan")
    return float(np.mean(np.linalg.norm(V - mu, axis=1)) / nmu)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leave_out", required=True, choices=PACS)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--datasetRoot", default="../datasets/")
    p.add_argument("--num_classes", type=int, default=6,
                   help="已知類別數（OSDG 6-way：person 已從訓練集排除）")
    p.add_argument("--exclude_class_idx", type=int, default=6,
                   help="ImageFolder 中 person 的 label；設 -1 表示不排除")
    p.add_argument("--num_nodes", type=int, default=9)
    p.add_argument("--randomSeed", type=int, default=2026,
                   help="須與訓練時一致——它同時是資料切分的 split_seed（util.py:810）")
    p.add_argument("--node_split_mode", default="class_balanced",
                   help="graphid=6 的預設值（train.py:2154）")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt_epoch", default="final")
    p.add_argument("--min_count_warn", type=int, default=30,
                   help="低於此樣本數的 (節點,類別) 格子標記為不可信")
    p.add_argument("--output_csv", required=True)
    args = p.parse_args()

    if args.exclude_class_idx is not None and args.exclude_class_idx < 0:
        args.exclude_class_idx = None

    device = args.device if torch.cuda.is_available() else "cpu"
    suffix = "final" if str(args.ckpt_epoch) == "final" else f"epoch_{args.ckpt_epoch}"

    node_to_domain, domain_to_nodes, node_subsets = rebuild_node_subsets(args)

    # ---------- 逐節點估原型 ----------
    protos, counts_all, low_count_cells = {}, {}, []
    for i in range(args.num_nodes):
        node = f"node_{i}"
        ckpt = os.path.join(args.checkpoint_dir, f"{args.description}_{node}_{suffix}.pth")
        if not os.path.exists(ckpt):
            print(f"[skip] missing {ckpt}")
            continue
        backbone, _ = load_backbone_diffusion(ckpt, args.num_classes, device)
        P, cnt = compute_prototypes(backbone, node_subsets[node], args.num_classes,
                                    device, args.batch_size, args.num_workers)
        protos[node] = P
        counts_all[node] = cnt
        for c in range(args.num_classes):
            if cnt[c] < args.min_count_warn:
                low_count_cells.append((node, c, int(cnt[c])))
        print(f"[proto] {node} ({node_to_domain[node]}): counts={cnt.tolist()}")
        del backbone

    if len(protos) < 2:
        print("Not enough nodes with checkpoints."); return

    # ---------- §0 自檢 ----------
    print("\n" + "=" * 70)
    print("§0 資料自檢")
    print("=" * 70)
    print(f"  節點數（有 checkpoint）: {len(protos)}/{args.num_nodes}")
    print(f"  每節點樣本數: "
          f"{ {n: int(counts_all[n].sum()) for n in sorted(protos)} }")
    if low_count_cells:
        print(f"  ⚠️ 樣本數 < {args.min_count_warn} 的格子（原型估計不可信、已排除於統計）:")
        for n, c, k in low_count_cells:
            print(f"      {n} class{c}: {k}")
    else:
        print(f"  ✅ 所有 (節點,類別) 格子樣本數 >= {args.min_count_warn}")

    # ---------- 兩兩夾角 ----------
    rows, within, cross = [], [], []
    cross_by_pair = defaultdict(list)
    nodes = sorted(protos.keys(), key=lambda s: int(s.split("_")[1]))

    for a, b in itertools.combinations(nodes, 2):
        da, db = node_to_domain[a], node_to_domain[b]
        same = (da == db)
        for c in range(args.num_classes):
            if counts_all[a][c] < args.min_count_warn or counts_all[b][c] < args.min_count_warn:
                continue
            ang = angle_deg(protos[a][c], protos[b][c])
            if np.isnan(ang):
                continue
            rows.append(dict(
                run=args.description, leave_out=args.leave_out,
                node_a=a, node_b=b, domain_a=da, domain_b=db,
                pair_type="within_domain" if same else "cross_domain",
                cls=c, angle_deg=round(ang, 4),
                n_a=int(counts_all[a][c]), n_b=int(counts_all[b][c])))
            (within if same else cross).append(ang)
            if not same:
                cross_by_pair[tuple(sorted([da, db]))].append(ang)

    # ---------- 判決 ----------
    print("\n" + "=" * 70)
    print(f"§1 判決  ({args.description})")
    print("=" * 70)
    if not within or not cross:
        print("  ⚠️ within 或 cross 樣本不足，無法判定");
    else:
        w_mean, c_mean = float(np.mean(within)), float(np.mean(cross))
        ratio = c_mean / w_mean if w_mean > 1e-9 else float("inf")
        print(f"  同域夾角（取樣噪聲地板） : {w_mean:8.3f}°  "
              f"[n={len(within)}, std={np.std(within):.3f}, "
              f"min={np.min(within):.3f}, max={np.max(within):.3f}]")
        print(f"  跨域夾角（主訊號）       : {c_mean:8.3f}°  "
              f"[n={len(cross)}, std={np.std(cross):.3f}, "
              f"min={np.min(cross):.3f}, max={np.max(cross):.3f}]")
        print(f"  ★ 比值 跨域/同域         : {ratio:8.3f}x")
        if ratio >= 3.0:
            print(f"\n  ✅ 失效證明成立（>= 3x）：原型確實按域分開，"
                  f"不聚合會讓各節點參照系不一致 ⇒ 進階段 1")
        elif ratio >= 1.5:
            print(f"\n  ⚠️ 中等訊號（1.5x ~ 3x）：有域效應但不強，"
                  f"調整一的必要性需再評估")
        else:
            print(f"\n  ❌ 跨域 ≈ 同域：原型不聚合**不構成問題**，"
                  f"調整一失去必要性 ⇒ 原型路線須重評")

    # ---------- 逐域對拆解 ----------
    if cross_by_pair:
        print(f"\n§2 逐域對拆解（跨域）")
        for pair, angs in sorted(cross_by_pair.items(), key=lambda kv: -np.mean(kv[1])):
            tag = ""
            if args.leave_out == "cartoon":
                ds = [DOMAIN_DIST_TO_CARTOON.get(d) for d in pair]
                if all(d is not None for d in ds):
                    tag = f"  (離 target 距離 {ds[0]}/{ds[1]})"
            print(f"  {pair[0]:14s} vs {pair[1]:14s}: {np.mean(angs):7.3f}°  "
                  f"[n={len(angs)}]{tag}")

    # ---------- 逐類別 ----------
    print(f"\n§3 逐類別（跨域夾角）")
    for c in range(args.num_classes):
        sub = [r["angle_deg"] for r in rows
               if r["cls"] == c and r["pair_type"] == "cross_domain"]
        subw = [r["angle_deg"] for r in rows
                if r["cls"] == c and r["pair_type"] == "within_domain"]
        if sub:
            wtxt = f"（同域 {np.mean(subw):6.3f}°）" if subw else ""
            print(f"  class{c}: 跨域 {np.mean(sub):7.3f}°  {wtxt}")

    # ---------- 原型分歧（對齊 BN-DIV 的尺） ----------
    print(f"\n§4 原型分歧 mean||v-mu||/||mu||（與 BN-DIV 同一把尺，0730 cartoon OFF = 0.345）")
    for c in range(args.num_classes):
        vecs = [protos[n][c] for n in nodes
                if counts_all[n][c] >= args.min_count_warn and not np.isnan(protos[n][c]).any()]
        if len(vecs) >= 2:
            print(f"  class{c}: {relative_divergence(vecs):.4f}  [n_nodes={len(vecs)}]")

    # ---------- 輸出 ----------
    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        write_header = not os.path.exists(args.output_csv)
        with open(args.output_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print(f"\nAppended {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
