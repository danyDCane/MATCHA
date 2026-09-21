"""檢測器有多「果斷」？——dany 2026-09-05 澄清：先不管判對判錯，只問判斷有沒有信心。

果斷 ＝ 分數遠離門檻、少有模稜兩可的樣本。不果斷 ＝ 一堆樣本擠在門檻附近，
翻一點點就改判。這與「判得準不準」是兩件獨立的事（可以很果斷地判錯）。

⚠️ 跨讀出比較必須無尺度：energy 約 −15~−5、原型角距離 0~90 度，|s−τ| 不可直接比。
   本腳本一律以【該節點來源域已知類別分數的標準差 σ_src】為單位——門檻本來就是從
   這條分布的 95 分位定出來的，用它的散布當尺規是自然的。

四個量：
  D1 分離度 Cohen's d ＝ (μ_未知 − μ_已知) / 併合標準差
     ⇒ 兩堆分開幾個標準差。無尺度、可跨讀出比。與 AUROC 單調相關但直接表達「分多開」。
  D2 果斷度 ＝ |s − τ| / σ_src 的中位數（混合流 π 加權）
     ⇒ 典型樣本離門檻幾個標準差。
  D3 模糊帶佔比 ＝ |s − τ| < 0.5·σ_src 的樣本比例
     ⇒ 有多少樣本是「翻一點點就改判」的。越低越果斷。
  D4 分方向的肯定度（已知類往「正常」推多遠、未知類往「異常」推多遠，皆以 σ_src 為單位）
     ⇒ 正值＝肯定且方向對；負值＝肯定地判錯那一側。
"""
import os
import sys
import numpy as np

PACS = ["art_painting", "cartoon", "photo", "sketch"]
SHORT = {"art_painting": "art", "cartoon": "cartoon", "photo": "photo", "sketch": "sketch"}
N, PI, Q = 9, 0.2, 0.95
COMBOS = [("baseline", "raw", "energy", "SOTA+energy/原樣"),
          ("baseline", "avg", "energy", "SOTA+energy/平均B"),
          ("ours", "raw", "energy", "我方+energy/原樣"),
          ("ours", "avg", "energy", "我方+energy/平均B"),
          ("ours", "raw", "proto", "我方+原型/原樣"),
          ("ours", "avg", "proto", "我方+原型/平均B")]


def nodes(fold, tag, bn, ro):
    z = np.load(f"results/osa/{fold}.npz", allow_pickle=True)
    out = []
    for i in range(N):
        p = f"{tag}__{bn}__{i}"
        out.append((z[f"{p}__src_{ro}"].astype(np.float64),
                    z[f"{p}__tgt_known_{ro}"].astype(np.float64),
                    z[f"{p}__tgt_unk_{ro}"].astype(np.float64)))
    return out


def stats(fold, tag, bn, ro):
    d1 = d2 = d3 = dk = du = 0.0
    for s_src, s_k, s_u in nodes(fold, tag, bn, ro):
        sig = s_src.std()
        tau = np.quantile(s_src, Q)
        pooled = np.sqrt((s_k.var() + s_u.var()) / 2)
        d1 += (s_u.mean() - s_k.mean()) / pooled / N
        dk_ = np.abs(s_k - tau) / sig
        du_ = np.abs(s_u - tau) / sig
        # π 加權的中位數：把兩堆的加權經驗分布合起來取 50% 位置
        allv = np.r_[dk_, du_]
        w = np.r_[np.full(len(dk_), (1 - PI) / len(dk_)), np.full(len(du_), PI / len(du_))]
        o = np.argsort(allv)
        cw = np.cumsum(w[o]) / w.sum()
        d2 += allv[o][np.searchsorted(cw, 0.5)] / N
        d3 += ((dk_ < 0.5).mean() * (1 - PI) + (du_ < 0.5).mean() * PI) / N
        dk += ((tau - s_k) / sig).mean() / N        # 已知類：往「正常」側推多遠
        du += ((s_u - tau) / sig).mean() / N        # 未知類：往「異常」側推多遠
    return d1, d2, d3, dk, du


R = {c[:3]: {f: stats(f, *c[:3]) for f in PACS} for c in COMBOS}
W = 96
for idx, (title, note) in enumerate([
        ("D1 分離度 Cohen's d（已知 vs 未知 分開幾個標準差）", "越大越好；0 ＝ 兩堆完全重疊"),
        ("D2 果斷度（典型樣本離門檻幾個 σ_src，π 加權中位數）", "越大越果斷"),
        ("D3 模糊帶佔比（|s−τ| < 0.5σ_src 的比例）", "越低越果斷；這些樣本翻一點點就改判"),
        ("D4a 已知類的肯定度（往『正常』側推幾個 σ）", "正＝肯定說正常；負＝肯定說異常（果斷判錯）"),
        ("D4b 未知類的肯定度（往『異常』側推幾個 σ）", "正＝肯定說異常")]):
    print("\n" + "=" * W)
    print(f"{title}")
    print("=" * W)
    print(f"  {note}")
    print(f"  {'組合':<22}" + "".join(f"{SHORT[f]:>11}" for f in PACS) + f"{'平均':>11}")
    for tag, bn, ro, name in COMBOS:
        v = [R[(tag, bn, ro)][f][idx] for f in PACS]
        print(f"  {name:<22}" + "".join(f"{x:>11.3f}" for x in v) + f"{np.mean(v):>11.3f}")
