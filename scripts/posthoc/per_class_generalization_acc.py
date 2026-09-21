#!/usr/bin/env python
"""泛化 ACC 的逐類別拆解（closed_acc by class）

問題：0905 §4.1 的「泛化」是目標域已知類六路 top-1，只有一個總數。
     本腳本把同一個量拆成六個已知類別，看是哪幾類在撐、哪幾類在拖。

⚠️ 不做任何前向：直接用 results/osa/<fold>.npz 已落盤的 `tgt_known_correct`（0/1），
   再用 ImageFolder 的 targets 重建同順序的標籤（loader shuffle=False ⇒ 順序可重現）。
   ⇒ 零 GPU、零重跑，且逐節點總平均必須與 §4.1 逐位吻合（腳本內自動對錨）。

聚合方式與 scripts/posthoc/osa_section4_tables.py 完全相同：
   逐節點算比例 → 9 節點算術平均（macro over node、micro over sample）。
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

PACS = ["art_painting", "cartoon", "photo", "sketch"]
N_NODES = 9
UNK = 6
CLS = ["dog", "elephant", "giraffe", "guitar", "horse", "house"]
CLS_ZH = ["狗", "大象", "長頸鹿", "吉他", "馬", "房子"]
ARMS = [("baseline", "raw", "StyleDDG／原樣"), ("baseline", "avg", "StyleDDG／平均B"),
        ("ours", "raw", "我方／原樣"), ("ours", "avg", "我方／平均B")]


def target_known_labels(root, dom):
    """重建目標域已知類標籤，順序＝ DataLoader(shuffle=False) 的順序。"""
    from torchvision.datasets import ImageFolder
    ds = ImageFolder(root=os.path.join(root, "PACS", dom))
    assert ds.classes[UNK] == "person", f"類別順序不符：{ds.classes}"
    assert list(ds.classes[:6]) == CLS, f"類別順序不符：{ds.classes}"
    lab = np.asarray(ds.targets)
    return lab[lab != UNK]


def per_class(z, lab, tag, bn):
    """回傳 (6 類準確率 %, 總體準確率 %)，皆為 9 節點平均。"""
    per_cls, overall = [], []
    for i in range(N_NODES):
        c = z[f"{tag}__{bn}__{i}__tgt_known_correct"] > 0
        assert c.shape == lab.shape, f"長度不符 {c.shape} vs {lab.shape}"
        per_cls.append([100 * c[lab == k].mean() for k in range(6)])
        overall.append(100 * c.mean())
    return np.mean(per_cls, axis=0), float(np.mean(overall))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default="results/osa")
    ap.add_argument("--datasetRoot", default="../datasets/")
    a = ap.parse_args()

    ANCHOR = {"art_painting": {"StyleDDG／原樣": 76.70, "StyleDDG／平均B": 84.68,
                               "我方／原樣": 82.58, "我方／平均B": 84.47},
              "cartoon":     {"StyleDDG／原樣": 71.53, "StyleDDG／平均B": 78.81,
                               "我方／原樣": 77.59, "我方／平均B": 79.20},
              "photo":       {"StyleDDG／原樣": 85.95, "StyleDDG／平均B": 91.03,
                               "我方／原樣": 87.28, "我方／平均B": 90.64},
              "sketch":      {"StyleDDG／原樣": 70.74, "StyleDDG／平均B": 71.06,
                               "我方／原樣": 71.59, "我方／平均B": 72.37}}

    store, ns = {}, {}
    for fold in PACS:
        z = np.load(os.path.join(a.npz_dir, f"{fold}.npz"))
        lab = target_known_labels(a.datasetRoot, fold)
        ns[fold] = [int((lab == k).sum()) for k in range(6)]
        for tag, bn, nm in ARMS:
            store[(fold, nm)] = per_class(z, lab, tag, bn)

    W = 9
    print("=" * 96)
    print("★ 泛化 ACC 的逐類別拆解｜目標域已知類六路 top-1、9 節點平均")
    print("   資料＝results/osa/<fold>.npz 的 tgt_known_correct（未重跑前向）")
    print("=" * 96)

    for fold in PACS:
        print(f"\n【{fold}】目標域已知類張數： "
              + "  ".join(f"{c}={n}" for c, n in zip(CLS_ZH, ns[fold]))
              + f"（合計 {sum(ns[fold])}）")
        print("  " + "臂".ljust(20) + "".join(c.rjust(W) for c in CLS_ZH)
              + "總體".rjust(W + 2) + "  §4.1錨點  差")
        for _, _, nm in ARMS:
            pc, ov = store[(fold, nm)]
            anc = ANCHOR[fold][nm]
            print("  " + nm.ljust(18) + "".join(f"{v:>{W}.2f}" for v in pc)
                  + f"{ov:>{W+2}.2f}" + f"{anc:>10.2f}" + f"{ov-anc:>+7.2f}")
        pc_o, _ = store[(fold, "我方／平均B")]
        pc_b, _ = store[(fold, "StyleDDG／原樣")]
        print("  " + "Δ 現行對外比法".ljust(17) + "".join(f"{o-b:>+{W}.2f}" for o, b in zip(pc_o, pc_b)))

    print("\n" + "=" * 96)
    print("★ 四折平均（先逐折算、再四折算術平均）")
    print("=" * 96)
    print("  " + "臂".ljust(20) + "".join(c.rjust(W) for c in CLS_ZH) + "總體".rjust(W + 2))
    avg = {}
    for _, _, nm in ARMS:
        m = np.mean([store[(f, nm)][0] for f in PACS], axis=0)
        o = np.mean([store[(f, nm)][1] for f in PACS])
        avg[nm] = m
        print("  " + nm.ljust(18) + "".join(f"{v:>{W}.2f}" for v in m) + f"{o:>{W+2}.2f}")
    d = avg["我方／平均B"] - avg["StyleDDG／原樣"]
    print("  " + "Δ 現行對外比法".ljust(17) + "".join(f"{v:>+{W}.2f}" for v in d))
    d2 = avg["我方／原樣"] - avg["StyleDDG／原樣"]
    print("  " + "Δ 同口徑都不匯聚".ljust(16) + "".join(f"{v:>+{W}.2f}" for v in d2))

    print("\n  逐類別勝場（我方平均B vs StyleDDG 原樣，四折）")
    for k, c in enumerate(CLS_ZH):
        w = [store[(f, "我方／平均B")][0][k] - store[(f, "StyleDDG／原樣")][0][k] for f in PACS]
        print(f"    {c:<6}" + "".join(f"{v:>+9.2f}" for v in w)
              + f"   平均{np.mean(w):>+7.2f}   勝{sum(1 for x in w if x > 0)}/4")


if __name__ == "__main__":
    main()
