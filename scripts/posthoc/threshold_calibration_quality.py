"""門檻校準資料的品質：分對了嗎？信心高嗎？分數擁擠嗎？（教授 2026-09-11 提問）

教授問了四件事，加上他自己的一個假設：
  Q1 用來算門檻（來源域 95 分位）的那批資料，本身都分類正確嗎？
  Q2 §3.4 表裡「放行分對」的那些圖，信心度高不高？
  Q3 來源域的分數會不會很擁擠（稍微動一下門檻就大量改判）？
  Q4 目標域的樣本是不是貼著門檻（稍微變動就從已知變未知、反之亦然）？
  H  教授的假設：「分類對但 logit 不高」的樣本會汙染門檻；
     「已知分對且高、未知都平低」才是好門檻。

⚠️ 方法紀律：不用 σ 當單位（σ 不是保序不變量，見 0905 §6.8）。
   「擁擠」一律用**決策單位**表達：把門檻從 q95 移到 q94／q96，改判幾個百分點。

★ H 用**介入**回答（不是相關性）：換掉校準集重算門檻，看部署指標怎麼變。
   臂A 全部來源域已知樣本（現行）｜臂B 只用分類正確的｜臂C 只用分類正確且最有把握的前 50%

用法：./venv_matcha/bin/python scripts/posthoc/threshold_calibration_quality.py
"""
import numpy as np

PACS = ["art_painting", "cartoon", "photo", "sketch"]
SH = {"art_painting": "art", "cartoon": "cartoon", "photo": "photo", "sketch": "sketch"}
N, Q, PI = 9, 0.95, 0.2
# 對外靶＝StyleDDG+energy／原樣；另列我方主組合對照
ARMS = [("StyleDDG+energy／原樣（對外靶）", "baseline", "raw", "energy"),
        ("我方+‖z⊥‖（面）／平均B（主組合）", "ours", "avg", "zperp")]


def node(fold, tag, bn, ro, i):
    z = np.load(f"results/osa/{fold}.npz", allow_pickle=True)
    p = f"{tag}__{bn}__{i}"
    return (z[f"{p}__src_{ro}"].astype(np.float64), z[f"{p}__src_correct"] > 0,
            z[f"{p}__tgt_known_{ro}"].astype(np.float64), z[f"{p}__tgt_known_correct"] > 0,
            z[f"{p}__tgt_unk_{ro}"].astype(np.float64))


def osa(k, c, u, tau):
    return 100 * ((1 - PI) * float(((k <= tau) & c).mean()) + PI * float((u > tau).mean()))


def main():
    W = 100
    for name, tag, bn, ro in ARMS:
        print("=" * W); print(f"★★ {name}   讀出＝{ro}"); print("=" * W)

        # ── Q1 校準資料本身的品質 ──
        print("\n【Q1】拿來算門檻的來源域資料，本身分類正確嗎？")
        print(f"  {'fold':<9}{'n':>7}{'分類正確率':>10}{'τ在分對樣本的分位':>18}{'τ附近±1%窗內分對率':>20}")
        for f in PACS:
            acc, qpos, near = [], [], []
            for i in range(N):
                s, sc, *_ = node(f, tag, bn, ro, i)
                tau = np.quantile(s, Q)
                acc.append(sc.mean())
                qpos.append((s[sc] < tau).mean())            # τ 落在「分對樣本」的第幾分位
                lo, hi = np.quantile(s, 0.94), np.quantile(s, 0.96)
                m = (s >= lo) & (s <= hi)
                near.append(sc[m].mean() if m.any() else np.nan)
            print(f"  {SH[f]:<9}{len(s):>7}{np.mean(acc)*100:>9.1f}%{np.mean(qpos)*100:>17.1f}%"
                  f"{np.nanmean(near)*100:>19.1f}%")

        # ── Q2 目標域三組的分數位置（以「離門檻幾個來源域分位」表示，無尺度選擇）──
        print("\n【Q2】目標域已知類三組 + 未知類，分數落在來源域的第幾分位（門檻＝95）")
        print(f"  {'fold':<9}{'誤拒(已知)':>11}{'放行分錯':>10}{'放行分對':>10}{'未知:拒絕':>11}{'未知:放行':>11}")
        for f in PACS:
            cols = [[] for _ in range(5)]
            for i in range(N):
                s, sc, k, kc, u = node(f, tag, bn, ro, i)
                tau = np.quantile(s, Q)
                pct = lambda v: 100 * np.searchsorted(np.sort(s), v) / len(s)
                g = [k[k > tau], k[(k <= tau) & ~kc], k[(k <= tau) & kc], u[u > tau], u[u <= tau]]
                for j, v in enumerate(g):
                    cols[j].append(np.median(pct(v)) if len(v) else np.nan)
            print(f"  {SH[f]:<9}" + "".join(f"{np.nanmean(c):>10.1f}%" for c in cols[:3])
                  + "".join(f"{np.nanmean(c):>10.1f}%" for c in cols[3:]))
        print("  （中位數。>95 ＝ 會被拒絕那一側；數字越貼近 95 代表越貼著門檻）")

        # ── Q3/Q4 擁擠度：門檻移動 ±1 分位，改判幾 % ──
        print("\n【Q3+Q4】把門檻從 q95 移到 q94／q96，各有多少比例改判（決策單位，無尺度選擇）")
        print(f"  {'fold':<9}{'來源域':>9}{'目標已知':>10}{'目標未知':>10}{'混合流(π加權)':>14}")
        for f in PACS:
            r = [[] for _ in range(4)]
            for i in range(N):
                s, sc, k, kc, u = node(f, tag, bn, ro, i)
                t, lo, hi = (np.quantile(s, q) for q in (Q, 0.94, 0.96))
                fl = lambda v: float((((v > lo) != (v > t)) | ((v > hi) != (v > t))).mean())
                r[0].append(fl(s)); r[1].append(fl(k)); r[2].append(fl(u))
                r[3].append(fl(k) * (1 - PI) + fl(u) * PI)
            print(f"  {SH[f]:<9}" + "".join(f"{np.mean(x)*100:>9.1f}%" for x in r))

        # ── H 介入：換校準集 ──
        print("\n【H・介入】換掉門檻的校準集，部署指標怎麼變（教授的假設直接驗證）")
        print(f"  {'fold':<9}{'臂':<26}{'誤拒率':>9}{'放行率':>9}{'OSA':>8}{'ΔOSA':>8}")
        for f in PACS:
            base = None
            for lbl, sel in [("A 全部來源域樣本（現行）", "all"),
                             ("B 只用分類正確的", "corr"),
                             ("C 分類正確且最有把握前50%", "conf")]:
                fp, ps, os_ = [], [], []
                for i in range(N):
                    s, sc, k, kc, u = node(f, tag, bn, ro, i)
                    if sel == "all":
                        cal = s
                    elif sel == "corr":
                        cal = s[sc]
                    else:
                        c2 = s[sc]
                        cal = c2[c2 <= np.median(c2)]      # 拒絕分數低＝更像 ID＝更有把握
                    tau = np.quantile(cal, Q)
                    fp.append(float((k > tau).mean())); ps.append(float((u <= tau).mean()))
                    os_.append(osa(k, kc, u, tau))
                m = np.mean(os_)
                if base is None:
                    base = m
                print(f"  {SH[f] if sel=='all' else '':<9}{lbl:<26}"
                      f"{np.mean(fp):>9.4f}{np.mean(ps):>9.4f}{m:>8.2f}{m-base:>+8.2f}")
        print()


if __name__ == "__main__":
    main()
