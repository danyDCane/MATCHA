"""門檻捨棄／誤拒的那批樣本，信心度與分類正確率如何？（教授 2026-09-11 提問 → dany 2026-09-14 擴充）

兩張同構的表：
  §5.2 來源域：門檻＝95 分位 ⇒ 設計上放棄 5%。那 5% 是隨機犧牲還是本來就可疑？
  §5.3 目標域：門檻仍來自來源域（部署時只能這樣訂），套到沒看過的畫風上會變成什麼樣？

資料來源（2026-09-17 改）：直接讀 `scripts/open_set_accuracy.py --stage infer` 產出的 results/osa/<fold>.npz，
  與 0905 §3 逐張表是**同一份判決**。舊版對每個 (fold, node, arm) 重跑 GPU 前向；存檔已含每張圖的
  最大類別機率（`*_msp`，定向為 −max softmax）與分類對錯，以 --agg node 重算與 0914 log 吻合 ⇒ 不再重跑前向。

彙總口徑 --agg（2026-09-17 改預設為 pool）：
  pool ＝每張圖算一次：9 節點的樣本倒進同一池，再算該群的平均信心度／正確率。
  node ＝舊版：各節點先算該群的平均信心度／正確率，再 9 節點等權（nanmean）。只為重現 0914 log 保留。
  ⚠️ 兩者只在「條件量」（某群的正確率、某群的平均信心度）不同；「各群佔比」與「每節點平均張數」兩者相同。
     node 讓「該群只有 10 張」的節點與「該群有 82 張」的節點同權 ⇒ 漏放 person 的信心度被高估
     （sketch 我方：node 0.824 vs pool 0.770），且某節點該群為 0 張時無定義
     （對外靶 sketch 節點 6/7/8 一張 person 都沒漏 ⇒ node 口徑那格只平均了 6 個節點）。
  四 fold 平均列：pool 的條件量以「該群四 fold 的佔比」加權（＝四個 1000 張倒進同一池，與 0905 §3.5 一致）；
     node 維持舊版的四 fold 直接等權。

用法：./venv_matcha/bin/python scripts/posthoc/threshold_tail_confidence.py [--agg pool|node]   （CPU，數秒）
"""
import argparse
import numpy as np

PACS = ["art_painting", "cartoon", "photo", "sketch"]
SH = {"art_painting": "art", "cartoon": "cartoon", "photo": "photo", "sketch": "sketch"}
N, QS = 9, [0.94, 0.95, 0.96]
ARMS = [("StyleDDG+energy／原樣", "baseline", "raw", "energy"),
        ("我方+‖z⊥‖面／平均B", "ours", "avg", "zperp")]


def load(npz_dir):
    """每個 (arm, fold) → 9 節點的陣列：來源域已知類 s_*、目標域已知類 k_*、目標域 person u_*。"""
    cache = {}
    for f in PACS:
        z = np.load(f"{npz_dir}/{f}.npz", allow_pickle=True)
        for nm, tag, bn, ro in ARMS:
            nodes = []
            for i in range(N):
                p = f"{tag}__{bn}__{i}"
                nodes.append(dict(
                    s_S=z[f"{p}__src_{ro}"], s_pm=-z[f"{p}__src_msp"].astype(np.float64),
                    s_c=z[f"{p}__src_correct"] > 0,
                    k_S=z[f"{p}__tgt_known_{ro}"], k_pm=-z[f"{p}__tgt_known_msp"].astype(np.float64),
                    k_c=z[f"{p}__tgt_known_correct"] > 0,
                    u_S=z[f"{p}__tgt_unk_{ro}"], u_pm=-z[f"{p}__tgt_unk_msp"].astype(np.float64)))
            cache[(nm, f)] = nodes
    return cache


def group(vals, masks, agg):
    """一個群在 9 節點上的條件平均。vals/masks：逐節點陣列。空群回 nan。"""
    if agg == "node":
        return float(np.nanmean([v[m].mean() if m.any() else np.nan for v, m in zip(vals, masks)]))
    tot = sum(int(m.sum()) for m in masks)
    return sum(float(v[m].sum()) for v, m in zip(vals, masks)) / tot if tot else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default="results/osa")
    ap.add_argument("--agg", choices=["pool", "node"], default="pool")
    a = ap.parse_args()
    CACHE = load(a.npz_dir)
    W = 118
    print(f"彙總口徑：{a.agg}（pool＝每張圖算一次；node＝節點等權，舊版）")

    print("#" * W); print("### 表一｜【來源域】門檻捨棄的那 5% 是什麼樣的樣本"); print("#" * W)
    for arm, *_ in ARMS:
        print("\n" + "=" * W); print(f"★★ {arm}"); print("=" * W)
        for q in QS:
            print(f"\n  ── 門檻 ＝ 來源域第 {q*100:.0f} 分位（設計上捨棄 {(1-q)*100:.0f}%）──")
            print(f"    {'fold':<9}{'捨棄n':>7}{'捨棄群softmax':>14}{'捨棄群正確率':>13}"
                  f"{'保留群softmax':>14}{'保留群正確率':>13}{'門檻窄帶softmax':>16}")
            for f in PACS:
                nodes = CACHE[(arm, f)]
                dm, km, bm = [], [], []
                for d in nodes:
                    S = d["s_S"]
                    tau = np.quantile(S, q)
                    lo, hi = np.quantile(S, q - 0.005), np.quantile(S, q + 0.005)
                    dm.append(S > tau); km.append(S <= tau); bm.append((S >= lo) & (S <= hi))
                pm, c = [d["s_pm"] for d in nodes], [d["s_c"] for d in nodes]
                n = np.mean([m.sum() for m in dm])
                print(f"    {SH[f]:<9}{n:>7.0f}{group(pm, dm, a.agg):>14.3f}{group(c, dm, a.agg)*100:>12.1f}%"
                      f"{group(pm, km, a.agg):>14.3f}{group(c, km, a.agg)*100:>12.1f}%{group(pm, bm, a.agg):>16.3f}")
        print()

    print("\n" + "#" * W); print("### 表二｜【目標域＝未見畫風】同一張表，門檻仍是來源域分位"); print("#" * W)
    for arm, *_ in ARMS:
        print("\n" + "=" * W); print(f"★★ {arm}"); print("=" * W)
        for q in QS:
            print(f"\n  ── 門檻 ＝ 來源域第 {q*100:.0f} 分位 ──")
            print(f"    {'fold':<9}| {'已知類:誤拒':^28} | {'已知類:放行':^18} | {'person':^24}")
            print(f"    {'':<9}| {'實際%':>7}{'softmax':>9}{'正確率':>8}"
                  f" | {'softmax':>9}{'正確率':>8} | {'拒絕%':>7}{'拒softmax':>9}{'漏放softmax':>10}")
            rows = []
            for f in PACS:
                nodes = CACHE[(arm, f)]
                taus = [np.quantile(d["s_S"], q) for d in nodes]
                dm = [d["k_S"] > t for d, t in zip(nodes, taus)]
                km = [~m for m in dm]
                rm = [d["u_S"] > t for d, t in zip(nodes, taus)]
                lm = [~m for m in rm]
                kpm, kc, upm = [d["k_pm"] for d in nodes], [d["k_c"] for d in nodes], [d["u_pm"] for d in nodes]
                r = dict(fr=np.mean([m.mean() for m in dm]), rr=np.mean([m.mean() for m in rm]),
                         pd=group(kpm, dm, a.agg), cd=group(kc, dm, a.agg),
                         pk=group(kpm, km, a.agg), ck=group(kc, km, a.agg),
                         pu=group(upm, rm, a.agg), pl=group(upm, lm, a.agg))
                rows.append(r)
                print(f"    {SH[f]:<9}| {r['fr']*100:>6.1f}%{r['pd']:>9.3f}{r['cd']*100:>7.1f}%"
                      f" | {r['pk']:>9.3f}{r['ck']*100:>7.1f}% | {r['rr']*100:>6.1f}%"
                      f"{r['pu']:>9.3f}{r['pl']:>10.3f}")
            if a.agg == "node":
                m = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
            else:
                # 條件量以該群四 fold 的佔比加權（＝四個 1000 張倒進同一池）
                def wavg(k, w):
                    ws = np.array([w(r) for r in rows]); vs = np.array([r[k] for r in rows])
                    ok = ws > 0
                    return float((ws[ok] * vs[ok]).sum() / ws[ok].sum())
                m = dict(fr=np.mean([r["fr"] for r in rows]), rr=np.mean([r["rr"] for r in rows]),
                         pd=wavg("pd", lambda r: r["fr"]), cd=wavg("cd", lambda r: r["fr"]),
                         pk=wavg("pk", lambda r: 1 - r["fr"]), ck=wavg("ck", lambda r: 1 - r["fr"]),
                         pu=wavg("pu", lambda r: r["rr"]), pl=wavg("pl", lambda r: 1 - r["rr"]))
            print(f"    {'平均':<9}| {m['fr']*100:>6.1f}%{m['pd']:>9.3f}{m['cd']*100:>7.1f}%"
                  f" | {m['pk']:>9.3f}{m['ck']*100:>7.1f}% | {m['rr']*100:>6.1f}%"
                  f"{m['pu']:>9.3f}{m['pl']:>10.3f}")
        print()
    print("  ⚠️ 平均列：佔比為四 fold 等權；" + ("條件量四 fold 直接等權（舊版）。" if a.agg == "node"
          else "條件量以該群四 fold 佔比加權（＝四個 1000 張倒進同一池，與 0905 §3.5 一致）。"))


if __name__ == "__main__":
    main()
