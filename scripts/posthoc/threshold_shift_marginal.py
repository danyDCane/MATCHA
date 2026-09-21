"""門檻從 95 分位移到 94／96，實際翻掉哪些圖？（dany 2026-09-14）

⚠️ 方向（我們的分數是【拒絕分數】：越大越像未知，超過 τ 就拒絕）：
      q94 ⇒ τ 變小 ⇒ **更嚴**：拒絕更多 ⇒ 誤拒↑、漏放↓
      q96 ⇒ τ 變大 ⇒ **更鬆**：拒絕更少 ⇒ 誤拒↓、漏放↑

要回答的是「邊際那批圖是什麼」：
  ① 更嚴之後被**額外誤拒**的已知圖，它們原本分類對嗎？（分對＝純損失；分錯＝丟掉不虧）
  ② 更嚴之後**額外攔下**的未知圖有幾張？
  ③ 更鬆之後**救回來**的已知圖，分類對嗎？
  ④ 更鬆之後**多漏掉**的未知圖有幾張？
  ⑤ 邊際交換率：每多攔 1 張未知，要多誤殺幾張已知？

資料全部來自 results/osa/*.npz（分數與分類對錯都已落盤）⇒ 純 CPU、不需重跑前向。

用法：./venv_matcha/bin/python scripts/posthoc/threshold_shift_marginal.py
"""
import argparse

import numpy as np

PACS = ["art_painting", "cartoon", "photo", "sketch"]
SH = {"art_painting": "art", "cartoon": "cartoon", "photo": "photo", "sketch": "sketch"}
N, PI = 9, 0.2
ARMS = [("StyleDDG+energy／原樣", "baseline", "raw", "energy"),
        ("我方+‖z⊥‖面／平均B", "ours", "avg", "zperp")]


def per_node(fold, tag, bn, ro, i):
    z = np.load(f"results/osa/{fold}.npz", allow_pickle=True)
    p = f"{tag}__{bn}__{i}"
    return (z[f"{p}__src_{ro}"].astype(np.float64),
            z[f"{p}__tgt_known_{ro}"].astype(np.float64),
            z[f"{p}__tgt_known_correct"] > 0,
            z[f"{p}__tgt_unk_{ro}"].astype(np.float64))


def stats(fold, tag, bn, ro, agg="pool"):
    """回傳 dict：以「假想 1000 張（已知 800／未知 200）」為單位的邊際變化。

    agg 只影響「翻掉的圖裡原本分對的比率」（`*_dk_corr`，分母只有翻掉的那幾張）：
      pool（預設，2026-09-18 改）＝每張圖算一次，9 節點翻掉的圖倒進同一池。
      node ＝舊版，各節點先算比率再 9 節點等權（nanmean）。
    ⚠️ 各節點翻掉的張數差很多（photo 對外靶更嚴：2～35 張）⇒ node 讓翻 2 張的節點與翻 35 張的同權
       （該格 82.5% vs pool 75.9%）。其餘欄位（張數、邊際交換率、ΔOSA）分母相同，兩種口徑逐位一致。
    """
    a = {k: [] for k in ["fr95", "lk95", "osa95",
                         "S_dk", "S_dk_corr", "S_du", "S_osa",      # 更嚴（→q94）
                         "L_dk", "L_dk_corr", "L_du", "L_osa",      # 更鬆（→q96）
                         "S_n", "S_cn", "L_n", "L_cn"]}
    for i in range(N):
        s, k, c, u = per_node(fold, tag, bn, ro, i)
        t94, t95, t96 = (np.quantile(s, q) for q in (0.94, 0.95, 0.96))
        rej = lambda t: (k > t, u > t)
        rk95, ru95 = rej(t95); rk94, ru94 = rej(t94); rk96, ru96 = rej(t96)
        osa = lambda rk, ru: 100 * ((1 - PI) * float((~rk & c).mean()) + PI * float(ru.mean()))
        a["fr95"].append(rk95.mean() * 800); a["lk95"].append((~ru95).mean() * 200)
        a["osa95"].append(osa(rk95, ru95))
        # 更嚴：95 → 94
        add_k = rk94 & ~rk95                      # 額外被誤拒的已知
        add_u = ru94 & ~ru95                      # 額外攔下的未知
        a["S_dk"].append(add_k.mean() * 800)
        a["S_dk_corr"].append(c[add_k].mean() if add_k.any() else np.nan)
        a["S_n"].append(int(add_k.sum())); a["S_cn"].append(int(c[add_k].sum()))
        a["S_du"].append(add_u.mean() * 200)
        a["S_osa"].append(osa(rk94, ru94) - osa(rk95, ru95))
        # 更鬆：95 → 96
        back_k = rk95 & ~rk96                     # 救回來的已知
        lost_u = ru95 & ~ru96                     # 多漏掉的未知
        a["L_dk"].append(back_k.mean() * 800)
        a["L_dk_corr"].append(c[back_k].mean() if back_k.any() else np.nan)
        a["L_n"].append(int(back_k.sum())); a["L_cn"].append(int(c[back_k].sum()))
        a["L_du"].append(lost_u.mean() * 200)
        a["L_osa"].append(osa(rk96, ru96) - osa(rk95, ru95))
    out = {k: float(np.nanmean(v)) for k, v in a.items()}
    if agg == "pool":
        for pre in ("S", "L"):
            n = sum(a[f"{pre}_n"])
            out[f"{pre}_dk_corr"] = sum(a[f"{pre}_cn"]) / n if n else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", choices=["pool", "node"], default="pool",
                    help="翻掉的圖「原本分對」比率的彙總方式：pool＝每張算一次（預設）／node＝節點等權（舊版）")
    args = ap.parse_args()
    W = 104
    R = {n: {f: stats(f, t, b, r, args.agg) for f in PACS} for n, t, b, r in ARMS}
    print(f"彙總口徑：{args.agg}（只影響「其中分類正確」與「比率」兩欄）")
    for title, pre, desc in [
            ("① 更嚴（門檻 95→94 分位，τ 變小、拒絕更多）", "S",
             "額外誤拒的已知↑、額外攔下的未知↑"),
            ("② 更鬆（門檻 95→96 分位，τ 變大、拒絕更少）", "L",
             "救回的已知↑、多漏掉的未知↑")]:
        print("=" * W); print(f"★ {title}"); print(f"  {desc}｜單位：假想 1000 張（已知 800／未知 200）"); print("=" * W)
        for nm, *_ in ARMS:
            print(f"\n  ── {nm} ──")
            h = ("額外誤拒" if pre == "S" else "救回已知", "其中分類正確",
                 "額外攔下未知" if pre == "S" else "多漏掉未知")
            print(f"    {'fold':<9}{h[0]:>10}{h[1]:>12}{'(比率)':>9}{h[2]:>13}{'邊際交換率':>12}{'ΔOSA':>9}")
            for f in PACS:
                v = R[nm][f]
                dk, cr, du, do = v[f"{pre}_dk"], v[f"{pre}_dk_corr"], v[f"{pre}_du"], v[f"{pre}_osa"]
                ratio = dk / du if du > 1e-9 else np.nan
                print(f"    {SH[f]:<9}{dk:>10.1f}{dk*cr:>12.1f}{cr*100:>8.1f}%{du:>13.1f}"
                      f"{ratio:>11.1f}:1{do:>+9.2f}")
            vs = [R[nm][f] for f in PACS]
            m = lambda k: np.mean([v[k] for v in vs])
            dk, du = m(f"{pre}_dk"), m(f"{pre}_du")
            crs = np.array([v[f"{pre}_dk_corr"] for v in vs])
            if args.agg == "pool":       # 四 fold 以各自翻掉的張數加權（＝四個 1000 張倒進同一池）
                w = np.array([v[f"{pre}_dk"] for v in vs])
                cr = float((w * crs).sum() / w.sum())
            else:
                cr = float(np.mean(crs))
            print(f"    {'平均':<9}{dk:>10.1f}{dk*cr:>12.1f}{cr*100:>8.1f}%{du:>13.1f}"
                  f"{dk/du:>11.1f}:1{m(f'{pre}_osa'):>+9.2f}")
        print()

    print("=" * W); print("★ 現況（95 分位）對照，同樣以 1000 張為單位"); print("=" * W)
    print(f"  {'組合':<24}{'誤拒已知':>10}{'漏放未知':>10}{'OSA':>9}")
    for nm, *_ in ARMS:
        vs = [R[nm][f] for f in PACS]
        print(f"  {nm:<24}{np.mean([v['fr95'] for v in vs]):>10.1f}"
              f"{np.mean([v['lk95'] for v in vs]):>10.1f}{np.mean([v['osa95'] for v in vs]):>9.2f}")
    print("\n  邊際交換率讀法：更嚴時＝『每多攔 1 張未知，要多誤殺幾張已知』；")
    print("                  更鬆時＝『每多漏 1 張未知，可以救回幾張已知』。越大代表越不划算／越划算。")


if __name__ == "__main__":
    main()
