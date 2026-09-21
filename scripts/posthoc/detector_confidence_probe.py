"""檢測器自己的「信心度」有沒有用？——dany 2026-09-05 提問。

問題：分類頭有信心度（softmax 最大機率），那 energy 給出的「拒絕/放行」判斷本身
      有沒有一個信心度？會不會 energy 分數在某些樣本上根本不可靠？

三個可測的問法（本腳本逐一回答，全部用既有 npz 的逐樣本分數，不重新前向）：
  Q1 分數的絕對尺度可跨域信任嗎？
     量：目標域已知類的分數中位數，落在【來源域分布】的哪個分位。
     沒有漂移 ⇒ 應落在 50 分位；落在 90+ ⇒ 整條分布右移，門檻的語意已經變了。
  Q2 「離門檻多遠」能不能當信心度？
     量：把混合流樣本按 |s−τ| 分 5 箱，看每箱的決策錯誤率是否單調下降。
     單調 ⇒ 這個量有效，可以拿來排「哪些樣本該交人工」。
  Q3 讓檢測器對最沒把握的樣本棄權，混合流準確率能賺多少？
     量：棄權比例 0/10/20/30% 時的混合流準確率（棄權樣本不計分，等同交人工）。

⚠️ 節點間分歧（K 顆頭吵架）這條**不做**——TaskBoard 0829 已 killed：
   BN 平均後九節點 logit 餘弦 1.0000、分歧率 0.20/0.33%、獨立資訊＝零。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PACS = ["art_painting", "cartoon", "photo", "sketch"]
SHORT = {"art_painting": "art", "cartoon": "cartoon", "photo": "photo", "sketch": "sketch"}
N = 9
PI = 0.2
Q = 0.95


def load(fold, tag, bn, ro):
    z = np.load(f"results/osa/{fold}.npz", allow_pickle=True)
    out = []
    for i in range(N):
        p = f"{tag}__{bn}__{i}"
        if f"{p}__src_{ro}" not in z:
            return None
        out.append((z[f"{p}__src_{ro}"].astype(np.float64),
                    z[f"{p}__tgt_known_{ro}"].astype(np.float64),
                    z[f"{p}__tgt_unk_{ro}"].astype(np.float64),
                    z[f"{p}__tgt_known_correct"] > 0))
    return out


COMBOS = [("baseline", "raw", "energy", "SOTA+energy/原樣"),
          ("baseline", "avg", "energy", "SOTA+energy/平均B"),
          ("ours", "raw", "energy", "我方+energy/原樣"),
          ("ours", "avg", "energy", "我方+energy/平均B"),
          ("ours", "raw", "proto", "我方+原型/原樣"),
          ("ours", "avg", "proto", "我方+原型/平均B")]
W = 100

print("=" * W)
print("Q1  分數的絕對尺度可跨域信任嗎？（目標域已知類的中位數落在來源域分布的第幾分位）")
print("=" * W)
print("  沒有漂移 ⇒ 50 分位。愈接近 100 ⇒ 分布整體右移愈嚴重，來源域定的門檻語意已變。")
print(f"\n  {'組合':<22}" + "".join(f"{SHORT[f]:>11}" for f in PACS) + f"{'平均':>11}")
for tag, bn, ro, name in COMBOS:
    v = []
    for f in PACS:
        D = load(f, tag, bn, ro)
        v.append(np.mean([(s_src < np.median(s_k)).mean() * 100 for s_src, s_k, _, _ in D]))
    print(f"  {name:<22}" + "".join(f"{x:>10.1f}%" for x in v) + f"{np.mean(v):>10.1f}%")

print("\n" + "=" * W)
print("Q2  「離門檻多遠」能不能當檢測器的信心度？（混合流樣本按 |s−τ| 分 5 箱的錯誤率）")
print("=" * W)
print(f"  錯誤定義：已知類被拒絕、或未知類被放行。π={PI:.0%} 加權。箱 1 ＝ 離門檻最近（最沒把握）")
for tag, bn, ro, name in COMBOS:
    print(f"\n  ── {name} ──")
    print(f"  {'fold':<10}{'箱1(最近)':>11}{'箱2':>9}{'箱3':>9}{'箱4':>9}{'箱5(最遠)':>11}{'單調?':>8}")
    for f in PACS:
        D = load(f, tag, bn, ro)
        rates = np.zeros(5)
        for s_src, s_k, s_u, corr in D:
            tau = np.quantile(s_src, Q)
            d = np.r_[np.abs(s_k - tau), np.abs(s_u - tau)]
            err = np.r_[(s_k > tau), (s_u <= tau)].astype(float)
            wt = np.r_[np.full(len(s_k), (1 - PI) / len(s_k)), np.full(len(s_u), PI / len(s_u))]
            o = np.argsort(d)
            d, err, wt = d[o], err[o], wt[o]
            cw = np.cumsum(wt) / wt.sum()
            for b in range(5):
                m = (cw > b / 5) & (cw <= (b + 1) / 5)
                rates[b] += (err[m] * wt[m]).sum() / wt[m].sum() / N if m.any() else 0
        mono = "✅" if all(rates[i] >= rates[i + 1] for i in range(4)) else "✗"
        print(f"  {SHORT[f]:<10}" + "".join(f"{r:>10.3f}" for r in rates) + f"{mono:>8}")

print("\n" + "=" * W)
print("Q3  讓檢測器對最沒把握的樣本棄權（交人工），混合流準確率賺多少？")
print("=" * W)
print(f"  棄權＝把 |s−τ| 最小的 X% 樣本抽走不計分。π={PI:.0%}")
print(f"\n  {'組合':<22}{'fold':<10}{'棄0%':>9}{'棄10%':>9}{'棄20%':>9}{'棄30%':>9}{'棄30%−棄0%':>12}")
for tag, bn, ro, name in COMBOS:
    for f in PACS:
        D = load(f, tag, bn, ro)
        accs = np.zeros(4)
        for s_src, s_k, s_u, corr in D:
            tau = np.quantile(s_src, Q)
            d = np.r_[np.abs(s_k - tau), np.abs(s_u - tau)]
            ok = np.r_[((s_k <= tau) & corr), (s_u > tau)].astype(float)
            wt = np.r_[np.full(len(s_k), (1 - PI) / len(s_k)), np.full(len(s_u), PI / len(s_u))]
            o = np.argsort(-d)                      # 最有把握的排前面
            ok, wt = ok[o], wt[o]
            cw = np.cumsum(wt) / wt.sum()
            for j, keep in enumerate([1.0, 0.9, 0.8, 0.7]):
                m = cw <= keep
                accs[j] += (ok[m] * wt[m]).sum() / wt[m].sum() * 100 / N
        print(f"  {name if f == PACS[0] else '':<22}{SHORT[f]:<10}"
              + "".join(f"{a:>9.2f}" for a in accs) + f"{accs[3]-accs[0]:>+12.2f}")
