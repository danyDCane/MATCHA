"""面讀出的可視化（cartoon 折）：①分數分布圖（對外靶 baseline energy vs 我方面讀出）②扇形圖（我方）。

【由來】1014 dany 問：person 在特徵空間「不在外面」（0903 §4），面讀出要怎麼畫才不誤導？
  三條「不在外面」的證據量的是「到最近中心的角度」與「面外那一截的方向」；面讀出量的是
  「面外那一截的長度」⇒ 圖的座標軸必須就是讀出用的量，不能用壓成平面的降維圖。

【口徑】對齊看板 §A「★★ 現行靶」（cartoon、ep200 final、9 節點）：
  baseline ＝ StyleDDG（BN 原樣、energy）。逐樣本分數走 osdg_eval 同一條路徑
             （load_backbone_diffusion → score_and_predict）重算，快取成 npz。
  我方     ＝ 全套＋原型（logs/prototype_probe/0908_dump/full_cartoon.npz：BN 平均B），
             面讀出 ‖z⊥‖ ＝ 單位化 z 落在「六個 EMA 類別中心張成的面」之外的長度。
【錨點】9 節點 AUROC 平均須逐位重現：baseline energy 0.8016（0909_zperp_oscr_4fold.csv）、
        我方 zperp 0.8281（§A／P16）。對不上就不出圖。

扇形圖：扇區＝最靠近哪一個類別中心；半徑＝被六個中心解釋的長度 √(1−‖z⊥‖²)
       （與面讀出是同一個量的單調變換：越靠圓心越可疑）。⚠️ 扇區內的角度位置是隨機擺放、不帶資訊。

用法：venv_matcha/bin/python scripts/posthoc/plane_readout_figures.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import numpy as np
from sklearn.metrics import roc_auc_score

N, PERSON, K = 9, 6, 6
FOLD = "cartoon"
BASE_RUN = "v1_stage2_leave_cartoon_nodiff_osdg_excl_person_seed2026_topo1234"
OUR_DUMP = "logs/prototype_probe/0908_dump/full_cartoon.npz"
BASE_CACHE = "logs/prototype_probe/1014_baseline_energy_cartoon.npz"
OUT_DIR = "research/prototype_probe/1014_plane_readout_figs"
ANCHOR = {"baseline": 0.8016, "ours": 0.8281}
CLS_ZH = ["狗", "大象", "長頸鹿", "吉他", "馬", "房子"]      # ImageFolder 字母序，person=6
NULL_ZPERP = float(np.sqrt(1 - K / 128))                       # 隨機單位向量的期望值（0829 §0：0.9763）
KEEP = 0.95                                                    # 門檻：保留 95% 已知類別


def baseline_scores():
    """回傳 9 個節點的 (reject_score, label)；reject = −logsumexp(logits)，越大越像未知。"""
    if os.path.exists(BASE_CACHE):
        d = np.load(BASE_CACHE)
        return [(d[f"n{i}_rej"], d[f"n{i}_y"]) for i in range(N)]
    import torch
    import test_domain_ood_scores as TD
    from osdg_eval import load_backbone_diffusion
    from joint_eval_mixed_stream import score_and_predict
    loader = TD.load_pacs_test_data("../datasets/", FOLD, 128, 4)[0]
    out, save = [], {}
    for i in range(N):
        ck = f"exp_result_{BASE_RUN}/{BASE_RUN}_node_{i}_final.pth"
        bb, diff = load_backbone_diffusion(ck, 6, "cuda")
        with torch.no_grad():
            _, _, en, _, lab = score_and_predict(bb, diff, loader, [], "eps_mse", "cuda")
        save[f"n{i}_rej"], save[f"n{i}_y"] = -en, lab
        out.append((-en, lab))
        del bb, diff
    np.savez(BASE_CACHE, **save)
    return out


def our_scores():
    """回傳 9 個節點的 (‖z⊥‖, label, 最近中心編號)。"""
    d = np.load(OUR_DUMP, allow_pickle=True)
    out = []
    for i in range(N):
        z = d[f"n{i}_tgt_z"].astype(np.float64)
        u = z / np.linalg.norm(z, axis=1, keepdims=True)
        C = d[f"n{i}_Cema"].astype(np.float64)
        C = C / np.linalg.norm(C, axis=1, keepdims=True)
        B = np.linalg.svd(C.T, full_matrices=False)[0]          # 面的正交基（等價 pinv 投影）
        zp = np.sqrt(np.clip(1.0 - ((u @ B) ** 2).sum(1), 0, None))
        out.append((zp, d[f"n{i}_tgt_y"].astype(int), (u @ C.T).argmax(1)))
    return out


def summarize(name, nodes):
    """逐節點 AUROC／攔下率，再平均；另報 9 節點合併後的值（圖畫的是合併）。"""
    au, caught = [], []
    for s, y in nodes:
        pos = y == PERSON
        au.append(roc_auc_score(pos, s))
        t = np.quantile(s[~pos], KEEP)
        caught.append(float((s[pos] > t).mean()))
    S = np.concatenate([s for s, _ in nodes]); Y = np.concatenate([y for _, y in nodes])
    pos = Y == PERSON
    t_pool = float(np.quantile(S[~pos], KEEP))
    res = dict(auroc=float(np.mean(au)), caught=float(np.mean(caught)),
               auroc_pool=float(roc_auc_score(pos, S)), caught_pool=float((S[pos] > t_pool).mean()),
               t_pool=t_pool, n_known=int((~pos).sum()), n_person=int(pos.sum()))
    print(f"\n[{name}] 逐節點 AUROC：" + " ".join(f"{a:.4f}" for a in au))
    print(f"[{name}] 逐節點 攔下 person @保留95%已知：" + " ".join(f"{c:.3f}" for c in caught))
    print(f"[{name}] 9 節點平均 AUROC = {res['auroc']:.4f}（錨點 {ANCHOR[name]}）｜"
          f"合併後 AUROC = {res['auroc_pool']:.4f}")
    print(f"[{name}] 攔下率 9 節點平均 = {res['caught']:.4f}｜合併後 = {res['caught_pool']:.4f}"
          f"（合併門檻 {t_pool:.4f}）｜樣本 已知 {res['n_known']}／person {res['n_person']}")
    assert abs(round(res["auroc"], 4) - ANCHOR[name]) < 1e-9, f"{name} 錨點對不上，停止出圖"
    return res


def main():
    base = baseline_scores()
    ours = our_scores()
    rb = summarize("baseline", base)
    ro = summarize("ours", [(s, y) for s, y, _ in ours])

    zp = np.concatenate([s for s, _, _ in ours]); Y = np.concatenate([y for _, y, _ in ours])
    sec = np.concatenate([c for _, _, c in ours]); pos = Y == PERSON
    r = np.sqrt(np.clip(1 - zp ** 2, 0, None))
    print("\n[ours] 面外長度 ‖z⊥‖ 均值：已知 %.4f／person %.4f／隨機零點 %.4f"
          % (zp[~pos].mean(), zp[pos].mean(), NULL_ZPERP))
    print("[ours] 被解釋長度 √(1−‖z⊥‖²) 均值：已知 %.4f／person %.4f／隨機零點 %.4f"
          % (r[~pos].mean(), r[pos].mean(), np.sqrt(K / 128)))
    share_p = np.bincount(sec[pos], minlength=K) / pos.sum()
    share_k = np.bincount(sec[~pos], minlength=K) / (~pos).sum()
    print("[ours] 各扇區佔比（最靠近哪個中心）：")
    for k in range(K):
        print(f"   {CLS_ZH[k]:4s} 已知 {share_k[k]:6.1%}   person {share_p[k]:6.1%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # .ttc 只會註冊第一個字面（JP），字形涵蓋繁中；matplotlib 預設字型快取找不到 CJK
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False,
                         "font.size": 10})
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
    C_KNOWN, C_PERSON = "#2a78d6", "#eb6834"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 圖一：分數分布（左 baseline energy、右 我方面讀出；各自的 x 軸）──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [(axes[0], base, rb, "對外靶 baseline（StyleDDG）：energy 分數",
               "−energy（越大越像未知類別）", None),
              (axes[1], [(s, y) for s, y, _ in ours], ro, "我們：面讀出",
               "落在六原型張成的面之外的長度 ‖z⊥‖（越大越像未知類別）", NULL_ZPERP)]
    for ax, nodes, res, title, xl, null in panels:
        S = np.concatenate([s for s, _ in nodes]); Yp = np.concatenate([y for _, y in nodes]) == PERSON
        lo, hi = np.quantile(S, [0.001, 0.999])
        if null is not None:
            hi = max(hi, null + 0.01)
        bins = np.linspace(lo, hi, 60)
        for m, col, lab, n in [(~Yp, C_KNOWN, "cartoon 已知類別", res["n_known"]),
                               (Yp, C_PERSON, "cartoon person（未知）", res["n_person"])]:
            ax.hist(S[m], bins=bins, density=True, color=col, alpha=0.28, linewidth=0)
            ax.hist(S[m], bins=bins, density=True, histtype="step", color=col, linewidth=2,
                    label=f"{lab}　n={n}")
        # 參考線的說明放進圖例，避免標字壓到曲線
        ax.axvline(res["t_pool"], color=INK2, linestyle="--", linewidth=1.2,
                   label="門檻：保留 95% 已知（9 節點合併）")
        if null is not None:
            ax.axvline(null, color=MUTED, linestyle=":", linewidth=1.5,
                       label=f"隨機向量的期望值 {null:.3f}")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel(xl, color=INK2); ax.set_ylabel("密度（各堆各自正規化）", color=INK2)
        ax.text(0.02, 0.97,
                f"AUROC {res['auroc']:.4f}（9 節點平均）\n"
                f"保留 95% 已知時攔下 person {res['caught']:.1%}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), frameon=False, fontsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2)
    fig.text(0.01, 0.005, "cartoon 折、ep200、9 節點合併成一張圖（AUROC 與攔下率為逐節點算後平均）。"
             "baseline BN 原樣；我們 BN 平均B。兩邊 x 軸是不同的分數、刻度不可互比。",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p1 = os.path.join(OUT_DIR, "fig1_score_distribution.png")
    fig.savefig(p1, dpi=200, facecolor="white"); plt.close(fig)

    # ── 圖二：扇形圖（我方；左已知、右 person，同一座標）──
    rng = np.random.default_rng(2026)
    width = 2 * np.pi / K
    theta = sec * width + width / 2 + rng.uniform(-0.42, 0.42, size=len(sec)) * width
    r_t = float(np.sqrt(max(1 - ro["t_pool"] ** 2, 0)))
    r_null = float(np.sqrt(K / 128))
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.0), subplot_kw={"projection": "polar"})
    for ax, m, col, title, n in [(axes[0], ~pos, C_KNOWN, "cartoon 已知類別", ro["n_known"]),
                                 (axes[1], pos, C_PERSON, "cartoon person（未知）", ro["n_person"])]:
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.scatter(theta[m], r[m], s=3, color=col, alpha=0.18 if col == C_KNOWN else 0.35,
                   linewidths=0, rasterized=True)
        tt = np.linspace(0, 2 * np.pi, 361)
        ax.plot(tt, np.full_like(tt, r_t), color=INK2, linestyle="--", linewidth=1.2)
        ax.plot(tt, np.full_like(tt, r_null), color=MUTED, linestyle=":", linewidth=1.5)
        ax.fill_between(tt, 0, r_t, color=INK2, alpha=0.05, linewidth=0)
        # 扇區之間留有空隙（抖動只用扇區寬度的 ±42%）⇒ 標字放在扇區分界線上不會壓到點
        box = dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.9)
        ax.text(np.pi, r_t, "門檻", fontsize=8, color=INK2, ha="center", va="center", bbox=box)
        ax.text(np.pi, r_null, "隨機向量", fontsize=8, color=MUTED, ha="center", va="center", bbox=box)
        ax.set_rlabel_position(0)
        for k in range(K):
            ax.plot([k * width, k * width], [0, 1], color=GRID, linewidth=0.8)
        share = share_k if col == C_KNOWN else share_p
        ax.set_xticks(np.arange(K) * width + width / 2)
        ax.set_xticklabels([f"{CLS_ZH[k]}\n{share[k]:.0%}" for k in range(K)], color=INK2)
        ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], color=MUTED, fontsize=8)
        ax.grid(color=GRID, linewidth=0.6); ax.spines["polar"].set_color(GRID)
        ax.set_title(f"{title}　n={n}\n半徑均值 {r[m].mean():.3f}", color=INK, fontsize=11, pad=18)
    fig.text(0.01, 0.01,
             "扇區＝最靠近哪一個類別中心（標籤下方為落在該扇區的比例）；半徑＝被六個類別中心解釋的長度 "
             "√(1−‖z⊥‖²)，越靠圓心越像未知類別。\n灰底圓盤＝會被拒絕的區域（門檻：保留 95% 已知）。"
             "扇區內的角度位置為隨機擺放、不帶資訊。cartoon 折、ep200、BN 平均B、9 節點合併。",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    p2 = os.path.join(OUT_DIR, "fig2_fan.png")
    fig.savefig(p2, dpi=200, facecolor="white"); plt.close(fig)
    print(f"\n輸出：{p1}\n　　　{p2}")


if __name__ == "__main__":
    main()
