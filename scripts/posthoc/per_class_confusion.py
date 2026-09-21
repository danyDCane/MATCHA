#!/usr/bin/env python
"""泛化 ACC 的混淆矩陣（closed-set confusion，六路）

§4.1b 拆出「sketch 的狗只有 26.46%」，但 0/1 對錯答不出「錯成哪一類」。
2026-09-15 起 open_set_accuracy.py 的 infer 段落盤 `*_pred`／`tgt_known_label`
⇒ 本腳本零前向、直接讀 npz。

聚合：逐節點算列正規化混淆率 → 9 節點算術平均（與 §4.1／§4.1b 相同）。
對角線必須與 §4.1b 的逐類別準確率逐位吻合（腳本內自動對錨）。
"""
import os, sys, argparse
import numpy as np

PACS = ["art_painting", "cartoon", "photo", "sketch"]
N_NODES, UNK = 9, 6
CLS_ZH = ["狗", "大象", "長頸鹿", "吉他", "馬", "房子"]
ARMS = [("baseline", "raw", "StyleDDG／原樣"), ("baseline", "avg", "StyleDDG／平均B"),
        ("ours", "raw", "我方／原樣"), ("ours", "avg", "我方／平均B")]


def confusion(z, tag, bn):
    """回傳 (6x6 列正規化混淆率 %, 9 節點平均)。row=真實、col=預測。"""
    lab = z["tgt_known_label"]
    mats = []
    for i in range(N_NODES):
        pr = z[f"{tag}__{bn}__{i}__tgt_known_pred"]
        m = np.zeros((6, 6))
        for t in range(6):
            sel = lab == t
            for q in range(6):
                m[t, q] = 100 * (pr[sel] == q).mean()
        mats.append(m)
    return np.mean(mats, axis=0)


def unk_dest(z, tag, bn):
    """person（未知類）被強制六選一時落到哪（%，9 節點平均）。"""
    out = []
    for i in range(N_NODES):
        pr = z[f"{tag}__{bn}__{i}__tgt_unk_pred"]
        out.append([100 * (pr == q).mean() for q in range(6)])
    return np.mean(out, axis=0)


def show(m, title, note=""):
    print(f"\n  {title}{note}")
    print("    " + "真實＼預測".ljust(10) + "".join(c.rjust(9) for c in CLS_ZH))
    for t in range(6):
        row = "".join((f"[{m[t,q]:>6.2f}]" if t == q else f"{m[t,q]:>9.2f}") for q in range(6))
        print("    " + CLS_ZH[t].ljust(8) + row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default="results/osa")
    ap.add_argument("--folds", nargs="*", default=PACS)
    ap.add_argument("--arms", nargs="*", default=["StyleDDG／原樣", "我方／平均B"])
    a = ap.parse_args()

    print("=" * 96)
    print("★ 泛化 ACC 的混淆矩陣｜目標域已知類六路、列正規化(%)、9 節點平均")
    print("   [對角線] ＝ §4.1b 的逐類別準確率")
    print("=" * 96)

    for fold in a.folds:
        z = np.load(os.path.join(a.npz_dir, f"{fold}.npz"))
        lab = z["tgt_known_label"]
        print("\n" + "=" * 96)
        print(f"【{fold}】n=" + " ".join(f"{c}:{int((lab==t).sum())}" for t, c in enumerate(CLS_ZH)))
        for tag, bn, nm in ARMS:
            if nm not in a.arms:
                continue
            m = confusion(z, tag, bn)
            # 對錨：對角線 == per_class_generalization_acc 的值
            show(m, nm)
            worst = int(np.argmin(np.diag(m)))
            off = [(q, m[worst, q]) for q in range(6) if q != worst]
            off.sort(key=lambda x: -x[1])
            print(f"      → 最差類「{CLS_ZH[worst]}」{m[worst,worst]:.2f}%，主要錯向："
                  + "、".join(f"{CLS_ZH[q]} {v:.2f}%" for q, v in off[:3]))
            u = unk_dest(z, tag, bn)
            print("      → person（未知類）被強制六選一時落點："
                  + "、".join(f"{CLS_ZH[q]} {u[q]:.1f}%" for q in np.argsort(-u)[:3]))


if __name__ == "__main__":
    main()
