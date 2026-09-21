#!/usr/bin/env python
"""OSCR 曲線圖（四 fold ＋ 四折平均）

OSCR 報的是面積，但面積把「在哪個工作點贏、贏多少」壓成一個數。畫出曲線才看得到
形狀差異——例如低 FPR 區（實際部署會待的地方）誰比較高。

x 軸 FPR ＝ 被放行的未知類佔全部未知類的比例
y 軸 CCR ＝ 已知類裡「分類正確 AND 被放行」的比例（上限＝ closed_acc）

曲線構造與 `scripts/osdg_eval.py:compute_oscr` 逐行相同；
資料來自 `results/osa/<fold>.npz`（含 2026-09-15 新增的 *_pred／tgt_known_label）。
⚠️ 聚合方式必須與既有 OSCR 數字一致：**逐節點算曲線 → 內插到共同 FPR 網格 → 9 節點平均**；
四折平均再對四條 fold 曲線等權平均（積分是線性的 ⇒ 面積＝四折面積的算術平均，腳本內對錨）。

預設只畫「對外靶 vs 現行對外組合」兩條（對外用）。
`--all_readouts` 會另外畫含點讀出與自家 energy 的四條版本（內部誠實對照用）。
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACS = ["art_painting", "cartoon", "photo", "sketch"]
N_NODES, UNK = 9, 6
GRID = np.linspace(0.0, 1.0, 1001)
TRAPZ = getattr(np, "trapezoid", np.trapz)
AT = [0.05, 0.10, 0.20, 0.50]          # 部署真正會待的低 FPR 區

BASE = ("baseline", "raw", "energy", "StyleDDG + energy / BN raw  (external target)", "#d62728", "-")
OURS = ("ours", "avg", "zperp", "ours + ||z_perp|| (face) / BN avg-B", "#1f77b4", "-")
EXTRA = [
    ("ours", "avg", "proto",  "ours + prototype angle (point) / BN avg-B", "#1f77b4", "--"),
    ("ours", "avg", "energy", "ours + energy / BN avg-B  (internal ref)",  "#7f7f7f", ":"),
]
# §7.7 已登記的四折面積（逐位對錨）
ANCHOR = {
    ("baseline", "raw", "energy"): [0.6314, 0.6294, 0.7775, 0.6441],
    ("ours", "avg", "zperp"):      [0.7268, 0.7031, 0.8444, 0.6088],
    ("ours", "avg", "proto"):      [0.7210, 0.6948, 0.8229, 0.5325],
    ("ours", "avg", "energy"):     [0.6866, 0.7094, 0.8698, 0.6450],
}
ANCHOR_MEAN = {("baseline", "raw", "energy"): 0.6706, ("ours", "avg", "zperp"): 0.7208,
               ("ours", "avg", "proto"): 0.6928, ("ours", "avg", "energy"): 0.7277}


def oscr_curve(rej, pred, lab):
    """與 osdg_eval.compute_oscr 同構，但回傳整條曲線。"""
    known = lab != UNK
    n_k = max(int(known.sum()), 1)
    n_u = max(int((~known).sum()), 1)
    correct = (pred == lab) & known
    order = np.argsort(-(-rej), kind="stable")       # conf = -rej，高信心先接受
    ccr = np.concatenate([[0.0], np.cumsum(correct[order].astype(float)) / n_k])
    fpr = np.concatenate([[0.0], np.cumsum((~known[order]).astype(float)) / n_u])
    return fpr, ccr


def node_curves(z, tag, bn, key):
    """9 節點曲線內插到共同 FPR 網格後平均；同時回傳逐節點面積的平均。"""
    lab_k = z["tgt_known_label"].astype(np.int64)
    ys, areas = [], []
    for i in range(N_NODES):
        p = f"{tag}__{bn}__{i}"
        rej = np.concatenate([z[f"{p}__tgt_known_{key}"], z[f"{p}__tgt_unk_{key}"]]).astype(np.float64)
        pred = np.concatenate([z[f"{p}__tgt_known_pred"], z[f"{p}__tgt_unk_pred"]]).astype(np.int64)
        lab = np.concatenate([lab_k, np.full(len(z[f"{p}__tgt_unk_{key}"]), UNK)])
        f, c = oscr_curve(rej, pred, lab)
        areas.append(float(TRAPZ(c, f)))
        ys.append(np.interp(GRID, f, c))
    return np.mean(ys, axis=0), float(np.mean(areas))


def ceiling(z, tag, bn):
    return float(np.mean([(z[f"{tag}__{bn}__{i}__tgt_known_correct"] > 0).mean() for i in range(N_NODES)]))


def draw(ax, curves, ceils, title):
    for (y, area, lbl, col, ls) in curves:
        ax.plot(GRID, y, color=col, ls=ls, lw=2.0, label=f"{lbl}  (AUC {area:.4f})")
    for (cv, col, side) in ceils:
        ax.axhline(cv, color=col, lw=0.8, alpha=0.45)
        ax.text(0.985, cv + (0.012 if side > 0 else -0.038), f"ceiling = closed_acc {cv:.3f}",
                ha="right", fontsize=7.2, color=col)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("FPR  (unknown accepted / all unknown)", fontsize=9)
    ax.set_ylabel("CCR  (known correct AND accepted / all known)", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", fontsize=8.0, framealpha=0.92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default="results/osa")
    ap.add_argument("--out_dir", default="results/osa")
    ap.add_argument("--all_readouts", action="store_true",
                    help="另外輸出含點讀出與自家 energy 的四條版本（內部用）")
    a = ap.parse_args()

    curves = [BASE, OURS] + (EXTRA if a.all_readouts else [])
    suffix = "_allreadouts" if a.all_readouts else ""

    print("=" * 92)
    print("★ OSCR 曲線｜逐節點算 → 內插共同 FPR 網格 → 9 節點平均 → 四折等權平均")
    print("  對錨：平均曲線面積 vs §7.7 登記值")
    print("=" * 92)

    Y, A, C, maxdiff = {}, {}, {}, 0.0
    for fi, fold in enumerate(PACS):
        z = np.load(os.path.join(a.npz_dir, f"{fold}.npz"))
        print(f"\n【{fold}】")
        for tag, bn, key, lbl, col, ls in curves:
            y, area = node_curves(z, tag, bn, key)
            Y[(fold, tag, bn, key)] = y
            A[(fold, tag, bn, key)] = area
            anc = ANCHOR[(tag, bn, key)][fi]
            maxdiff = max(maxdiff, abs(area - anc))
            print(f"  {lbl:<48} 面積 {area:.4f}  §7.7 {anc:.4f}  差 {area-anc:+.4f}")
        C[(fold, "ours")] = ceiling(z, "ours", "avg")
        C[(fold, "baseline")] = ceiling(z, "baseline", "raw")
        yb, yo = Y[(fold, *BASE[:3])], Y[(fold, *OURS[:3])]
        d = yo - yb
        print("    低 FPR 工作點 CCR（面積藏起來的部分）")
        print("      " + "讀出".ljust(30) + "".join(f"FPR={t:.2f}".rjust(11) for t in AT))
        for tag, bn, key, lbl, _, _ in curves:
            y = Y[(fold, tag, bn, key)]
            print("      " + lbl.split("  ")[0].ljust(28)
                  + "".join(f"{np.interp(t, GRID, y):>11.4f}" for t in AT))
        print("      我方(面) − 對外靶：" + "  ".join(
            f"{t:.2f} {np.interp(t, GRID, d):+.4f}" for t in AT))
        sg = np.sign(d)
        cross = [GRID[i] for i in range(1, len(GRID)) if sg[i] and sg[i-1] and sg[i] != sg[i-1]]
        print("      交叉點：" + (", ".join(f"{c:.3f}" for c in cross) if cross else "全程同號、無交叉"))

    # ── 圖 1：四 fold ──
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
    for fi, fold in enumerate(PACS):
        cs = [(Y[(fold, t, b, k)], A[(fold, t, b, k)], lbl, col, ls)
              for t, b, k, lbl, col, ls in curves]
        draw(axes[fi // 2][fi % 2], cs,
             [(C[(fold, "ours")], "#1f77b4", +1), (C[(fold, "baseline")], "#d62728", -1)],
             f"leave-out: {fold}")
    fig.suptitle("OSCR curves (Dhamija et al., NeurIPS 2018) — PACS 4-fold, ep200, 9-node mean",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    o1 = os.path.join(a.out_dir, f"oscr_curves_4fold{suffix}.png")
    fig.savefig(o1, dpi=170); plt.close(fig)

    # ── 圖 2：四折平均 ──
    print("\n" + "=" * 92)
    print("★ 四折平均曲線")
    print("=" * 92)
    fig2, ax2 = plt.subplots(figsize=(7.4, 6.2))
    cs, dmean = [], {}
    for tag, bn, key, lbl, col, ls in curves:
        ym = np.mean([Y[(f, tag, bn, key)] for f in PACS], axis=0)
        am = float(np.mean([A[(f, tag, bn, key)] for f in PACS]))
        anc = ANCHOR_MEAN[(tag, bn, key)]
        maxdiff = max(maxdiff, abs(am - anc))
        print(f"  {lbl:<48} 面積 {am:.4f}  §7.7 {anc:.4f}  差 {am-anc:+.4f}"
              f"   （曲線積分 {TRAPZ(ym, GRID):.4f}）")
        cs.append((ym, am, lbl, col, ls))
        dmean[(tag, bn, key)] = ym
    cb = float(np.mean([C[(f, "baseline")] for f in PACS]))
    co = float(np.mean([C[(f, "ours")] for f in PACS]))
    draw(ax2, cs, [(co, "#1f77b4", +1), (cb, "#d62728", -1)],
         "PACS 4-fold mean  (equal weight per fold)")
    fig2.tight_layout()
    o2 = os.path.join(a.out_dir, f"oscr_curve_mean{suffix}.png")
    fig2.savefig(o2, dpi=170); plt.close(fig2)

    dm = dmean[OURS[:3]] - dmean[BASE[:3]]
    print("\n  四折平均｜我方(面) − 對外靶：" + "  ".join(
        f"FPR={t:.2f} {np.interp(t, GRID, dm):+.4f}" for t in AT))
    sg = np.sign(dm)
    cr = [GRID[i] for i in range(1, len(GRID)) if sg[i] and sg[i-1] and sg[i] != sg[i-1]]
    print("  四折平均｜交叉點：" + (", ".join(f"{c:.3f}" for c in cr) if cr else "全程同號、無交叉"))
    print(f"\n對錨最大差 {maxdiff:.4f}" + ("  ✅" if maxdiff < 5e-4 else "  ⚠️ 超過 0.0005，需查"))
    print(f"圖 → {o1}\n圖 → {o2}")


if __name__ == "__main__":
    main()
