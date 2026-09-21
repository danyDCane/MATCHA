"""果斷度（D2/D3）是不是尺度假象？——三個獨立檢驗（dany 2026-09-10 追問）

背景：0905 §6.5 記「原型讀出果斷度是 energy 的 1.9 倍、模糊帶只有一半，四 fold 全部一致」，
      並在 §7.6 把 MSP 的高果斷度標為「尺度假象」（softmax 飽和 ⇒ σ_src 極小 ⇒ |s−τ|/σ_src 膨脹）。
問題：原型讀出會不會踩到**同一個**機制？

D2/D3 都以 σ_src 為單位。σ **不是單調變換不變量**，而檢測器的判決（分數超過來源域 95 分位就拒絕）
**是**單調變換不變量 ⇒ 任何非單調不變的量都可能在量「分數怎麼寫」而不是「檢測器怎麼判」。

三個檢驗（全部不需要選單位，或用單調不變的單位）：
  T1 分位尺模糊帶：帶 = [來源域 q90, q99]（門檻 q95 在中間），每個讀出的來源域都恰有 9% 落在帶內
     ⇒ 同一把尺、且對任何單調變換完全免疫。
  T2 門檻重估改判率：bootstrap 重抽來源域分數 → 重算 95 分位 → 數部署流有多少判決翻掉。
     ⇒ 完全不需要選單位，直接就是「門檻穩不穩」。
  T3 單調變換示範：用來源域自己的累積分布把分數轉成常態，**門檻一起轉**。
     ⇒ 判決必定一張都不變（驗證 flips=0）；若 D2/D3 因此改變 ⇒ 它們量的不是判決。

輸入：results/osa/{fold}.npz
用法：./venv_matcha/bin/python scripts/posthoc/decisiveness_is_scale_artifact.py
"""
import numpy as np
from scipy.stats import norm

PACS = ["art_painting", "cartoon", "photo", "sketch"]
SH = {"art_painting": "art", "cartoon": "cartoon", "photo": "photo", "sketch": "sketch"}
N, PI, Q, B = 9, 0.2, 0.95, 400
COMBOS = [("baseline", "raw", "energy", "SOTA+energy"), ("ours", "raw", "energy", "我方+energy"),
          ("ours", "raw", "msp", "我方+MSP"), ("ours", "raw", "proto", "我方+原型"),
          ("ours", "raw", "zperp", "我方+面")]


def load(f, tag, bn, ro):
    z = np.load(f"results/osa/{f}.npz", allow_pickle=True)
    if f"{tag}__{bn}__0__src_{ro}" not in z:
        return None
    return [(z[f"{tag}__{bn}__{i}__src_{ro}"].astype(np.float64),
             z[f"{tag}__{bn}__{i}__tgt_known_{ro}"].astype(np.float64),
             z[f"{tag}__{bn}__{i}__tgt_unk_{ro}"].astype(np.float64)) for i in range(N)]


def wmix(k, u):
    return np.r_[np.full(len(k), (1 - PI) / len(k)), np.full(len(u), PI / len(u))]


def d2d3(s, k, u, tau):
    sig = s.std()
    dk, du = np.abs(k - tau) / sig, np.abs(u - tau) / sig
    allv, w = np.r_[dk, du], wmix(k, u)
    o = np.argsort(allv)
    cw = np.cumsum(w[o]) / w.sum()
    return allv[o][np.searchsorted(cw, 0.5)], (dk < 0.5).mean() * (1 - PI) + (du < 0.5).mean() * PI


def main():
    W = 96
    rng = np.random.default_rng(2026)
    R = {}
    for tag, bn, ro, name in COMBOS:
        acc = {k: [] for k in ["d2", "d3", "qband", "sband", "boot", "d2t", "d3t", "width", "skew", "kurt"]}
        flips = 0
        per_fold = {k: [] for k in ["qband", "boot"]}
        for f in PACS:
            D = load(f, tag, bn, ro)
            if D is None:
                acc = None
                break
            fq, fb = [], []
            for s, k, u in D:
                tau = np.quantile(s, Q)
                a, b = d2d3(s, k, u, tau)
                acc["d2"].append(a); acc["d3"].append(b)
                # T1 分位尺
                lo, hi = np.quantile(s, 0.90), np.quantile(s, 0.99)
                inb = lambda v: ((v >= lo) & (v <= hi)).mean()
                q = inb(k) * (1 - PI) + inb(u) * PI
                acc["qband"].append(q); acc["sband"].append(inb(s)); fq.append(q)
                acc["width"].append((hi - lo) / s.std())
                m, sd = s.mean(), s.std()
                acc["skew"].append(((s - m) ** 3).mean() / sd ** 3)
                acc["kurt"].append(((s - m) ** 4).mean() / sd ** 4 - 3)
                # T2 bootstrap 門檻重估改判率
                bk, bu = k > tau, u > tau
                taus = np.quantile(rng.choice(s, (B, len(s)), replace=True), Q, axis=1)
                fk = np.mean([((k > t) != bk).mean() for t in taus])
                fu = np.mean([((u > t) != bu).mean() for t in taus])
                bt = fk * (1 - PI) + fu * PI
                acc["boot"].append(bt); fb.append(bt)
                # T3 單調變換（門檻一起轉 ⇒ 判決必不變）
                ss, n = np.sort(s), len(s)
                T = lambda v: norm.ppf(np.clip(np.interp(v, ss, (np.arange(n) + 0.5) / n), 1e-6, 1 - 1e-6))
                a2, b2 = d2d3(T(s), T(k), T(u), T(tau))
                acc["d2t"].append(a2); acc["d3t"].append(b2)
                flips += int(((k > tau) != (T(k) > T(tau))).sum() + ((u > tau) != (T(u) > T(tau))).sum())
            if acc is None:
                break
            per_fold["qband"].append(np.mean(fq)); per_fold["boot"].append(np.mean(fb))
        if acc is not None:
            R[name] = ({k: float(np.mean(v)) for k, v in acc.items()}, flips, per_fold)

    print("=" * W); print("★ T1｜把模糊帶改用【來源域分位數】定義（對任何單調變換免疫）"); print("=" * W)
    print("  帶 = [來源域 q90, q99]，門檻 q95 在中間 ⇒ 每個讀出的來源域都恰有 9% 在帶內（同一把尺）")
    print(f"\n  {'組合':<16}{'D3(σ尺)':>10}{'分位尺':>9}" + "".join(f"{SH[f]:>9}" for f in PACS))
    for n, (v, _, pf) in R.items():
        print(f"  {n:<16}{v['d3']:>10.3f}{v['qband']:>9.3f}" + "".join(f"{x:>9.3f}" for x in pf["qband"]))
    print("\n  ⇒ σ 尺與分位尺**名次相反**：σ 尺說原型最果斷，分位尺說原型最不果斷。")

    print("\n" + "=" * W); print("★ T2｜門檻重估改判率（bootstrap 400 次，完全不需要選單位）"); print("=" * W)
    print(f"\n  {'組合':<16}{'平均':>9}" + "".join(f"{SH[f]:>9}" for f in PACS))
    for n, (v, _, pf) in R.items():
        print(f"  {n:<16}{v['boot']:>9.4f}" + "".join(f"{x:>9.4f}" for x in pf["boot"]))
    print("\n  ⇒ 原型並沒有比 energy 穩，反而略差（3/4 fold）。")

    print("\n" + "=" * W); print("★ T3｜單調變換示範（門檻一起轉 ⇒ 判決一張都不變）"); print("=" * W)
    print(f"\n  {'組合':<16}{'D2原始':>9}{'D2轉後':>9}{'D3原始':>9}{'D3轉後':>9}{'判決改變':>9}")
    for n, (v, fl, _) in R.items():
        print(f"  {n:<16}{v['d2']:>9.3f}{v['d2t']:>9.3f}{v['d3']:>9.3f}{v['d3t']:>9.3f}{fl:>9}")
    print("\n  ⇒ 判決改變全為 0，但 D2/D3 名次完全重排 ⇒ **D2/D3 量的是分數怎麼寫，不是檢測器怎麼判**。")

    print("\n" + "=" * W); print("★ 機制｜來源域分數分布的形狀（為什麼 σ 尺會偏心）"); print("=" * W)
    print("  常態分布時 q90→q99 ≈ 1.05σ，與 D3 的 ±0.5σ（寬 1.0σ）幾乎等寬")
    print(f"\n  {'組合':<16}{'q90→q99幾個σ':>14}{'/1.05':>8}{'偏度':>9}{'峰度':>9}")
    for n, (v, _, _) in R.items():
        print(f"  {n:<16}{v['width']:>14.2f}{v['width']/1.05:>8.2f}{v['skew']:>9.2f}{v['kurt']:>9.2f}")
    print("\n  ⇒ 原型的來源域分布右偏（偏度 +1.27）、尾巴長 ⇒ σ 被尾巴撐大 ⇒ ±0.5σ 這條帶在分位上很窄")
    print("     ⇒ 落在帶內的樣本自然少 ⇒ D3 看起來低。這與 MSP 的假象是**同一個機制**，只是程度較輕。")


def arccos_vs_cos():
    """T4｜我們自己 code 裡的一個任意選擇：`detection_score` 最後做了 arccos。

    直接用 −cos 當分數，排序完全相同 ⇒ 同一批圖被拒絕。若 D2/D3 因此改變，
    代表這個數字取決於「我們順手怎麼寫」，不是檢測器的性質。
    """
    W = 96
    print("\n" + "=" * W)
    print("★ T4｜arccos vs cos：`dood/prototype.py:detection_score` 最後那一步 arccos 是任意的")
    print("=" * W)
    A2 = A3 = B2 = B3 = 0.0
    n = flips = 0
    for f in PACS:
        D = load(f, "ours", "raw", "proto")
        for s, k, u in D:                                   # s,k,u：角度（現行）
            tau = np.quantile(s, Q)
            a, b = d2d3(s, k, u, tau); A2 += a; A3 += b
            # 餘弦版：−cos。arccos 遞減 ⇒ −cos 與角度同序 ⇒ 判決必定相同
            S, K, U = (-np.cos(np.radians(x)) for x in (s, k, u))
            T = -np.cos(np.radians(tau))
            a2, b2 = d2d3(S, K, U, T); B2 += a2; B3 += b2
            flips += int(((k > tau) != (K > T)).sum() + ((u > tau) != (U > T)).sum())
            n += 1
    print(f"\n  {'分數怎麼寫':<16}{'果斷度 D2':>12}{'模糊帶 D3':>12}")
    print(f"  {'角度（現行）':<16}{A2/n:>12.3f}{A3/n:>12.3f}")
    print(f"  {'餘弦':<16}{B2/n:>12.3f}{B3/n:>12.3f}")
    print(f"\n  兩版判決不同的圖：{flips} 張")
    print(f"  ⇒ 同一個檢測器、同一批判決，D3 可以是 {min(A3,B3)/n:.3f} 也可以是 {max(A3,B3)/n:.3f}。")
    print(f"     加上 T3 的常態版（0.360），這個數字的可選範圍是 0.163～0.360。")


if __name__ == "__main__":
    main()
    arccos_vs_cos()
