"""點讀出 → 面讀出：改善到底來自「已知少誤拒」還是「未知多抓到」？（dany 2026-10-01 提問）

背景：0905 §4.4 只報了 AUROC（0.7901 → 0.8349），§4.2 只報了誤拒率（0.2545 → 0.2452）。
      AUROC 是全域平均、不對應任何操作點；誤拒率只看已知那一側。
      ⇒ 兩邊都無法回答「換讀出之後，實際上是哪一側變好」。

本檔在**同一個操作點**（門檻＝來源域已知分數的 0.95 分位，與全專案一致）上，
把兩個讀出的四格去向並排，直接拆成兩側：
  已知側：誤拒率  ＝ (rej_ok + rej_wrong)          ← 越低越好
  未知側：正確拒絕率 ＝ unk_rej                     ← 越高越好

另外回答幾何上的困惑：「怎麼可能點的距離近、面的距離反而遠？」
  ⇒ **不可能**。六個類別中心都在那個面上 ⇒ 對任何樣本，
     到面的距離（Euclidean）≤ 到最近中心的弦距離 = 2·sin(θ_min/2)。
     換成面之後**每個樣本的分數都只會變小、不會變大**（【3】實測違反數 = 0／四折全部樣本）。
  ⛔ **原本寫的假設「已知那一側掉得比未知多」已被本檔數據推翻**（2026-10-01）：
     實測未知縮水 0.21–0.24、已知只縮 0.09–0.21，**未知反而縮得多**，均值差也變小。
  ✅ **真正的機制是散佈變窄**：點讀出的分數取決於「恰好哪一個中心最近」，這個挑選很不穩定
     ⇒ 樣本之間的分數散很開；面讀出不挑，永遠對同一個固定參照物量 ⇒ 量出來穩定得多。
     合併標準差縮得比均值差更兇（cartoon ×0.67、sketch ×0.59）⇒ 標準化分離度 d 上升 ⇒ 排序變好。
     【4】段同時列 d 與 AUROC（AUROC 可與 0905 §4.4 逐位對錨）。

口徑（與 0905 全檔一致）：ours 臂、BN 平均B、四折、9 節點判決倒進同一池（每張圖算一次）。
用法：./venv_matcha/bin/python scripts/posthoc/point_vs_surface_which_side.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from osa_4fold_summary import CELLS, cell_pool, N_NODES  # noqa: E402

PACS = ["art_painting", "cartoon", "photo", "sketch"]
SHORT = {"art_painting": "art", "cartoon": "cartoon", "photo": "photo", "sketch": "sketch"}
NPZ_DIR = "results/osa"
TAG, BN, Q = "ours", "avg", 0.95


def sides(cells):
    """四格 → 兩側的率。"""
    return (cells["rej_ok"][0] + cells["rej_wrong"][0],   # 已知誤拒率
            cells["unk_rej"][0])                          # 未知正確拒絕率


def main():
    print(f"口徑：{TAG} 臂／BN {BN}（平均B）／門檻＝來源已知分數 {Q} 分位／每張圖算一次\n")

    print("【1】同一個操作點上，兩個讀出的兩側表現")
    print(f"  {'fold':<9}{'已知誤拒率 點':>14}{'面':>9}{'改善':>9}   "
          f"{'未知正確拒絕率 點':>18}{'面':>9}{'改善':>9}")
    acc = {"proto": [[], []], "zperp": [[], []]}
    for d in PACS:
        z = np.load(f"{NPZ_DIR}/{d}.npz", allow_pickle=True)
        r = {}
        for ro in ("proto", "zperp"):
            fr, tr = sides(cell_pool(z, TAG, BN, ro, q=Q))
            r[ro] = (fr, tr)
            acc[ro][0].append(fr)
            acc[ro][1].append(tr)
        print(f"  {SHORT[d]:<9}{r['proto'][0]:>13.1%}{r['zperp'][0]:>9.1%}"
              f"{r['proto'][0]-r['zperp'][0]:>+9.1%}   "
              f"{r['proto'][1]:>17.1%}{r['zperp'][1]:>9.1%}{r['zperp'][1]-r['proto'][1]:>+9.1%}")
    fp, tp = np.mean(acc["proto"][0]), np.mean(acc["proto"][1])
    fz, tz = np.mean(acc["zperp"][0]), np.mean(acc["zperp"][1])
    print(f"  {'四折平均':<7}{fp:>13.1%}{fz:>9.1%}{fp-fz:>+9.1%}   "
          f"{tp:>17.1%}{tz:>9.1%}{tz-tp:>+9.1%}")
    print(f"\n  ⇒ 已知側（少誤拒）貢獻 {fp-fz:+.1%}／未知側（多抓到）貢獻 {tz-tp:+.1%}")

    print("\n【2】換算成每 1000 張（已知 800 ＋ 未知 200，π=20%）的張數變化")
    print(f"  已知被誤拒：{fp*800:>6.1f} 張 → {fz*800:>6.1f} 張　（{(fz-fp)*800:+.1f} 張）")
    print(f"  未知被抓到：{tp*200:>6.1f} 張 → {tz*200:>6.1f} 張　（{(tz-tp)*200:+.1f} 張）")

    print("\n【3】幾何驗證：到面的距離是否恆 ≤ 到最近中心的弦距離？")
    print("     （proto 存的是角距離，單位是『度』——open_set_accuracy.py:128 乘了 DEG；")
    print("      弦距離 = 2·sin(θ/2)，與 zperp 同為單位球上的歐氏長度，可直接比大小）")
    print(f"  {'fold':<9}{'群':<10}{'點(弦距)':>11}{'面(殘差)':>11}{'縮水':>9}{'違反數':>8}")
    for d in PACS:
        z = np.load(f"{NPZ_DIR}/{d}.npz", allow_pickle=True)
        for grp, lab in [("tgt_known", "已知"), ("tgt_unk", "未知")]:
            ch, rs, bad = [], [], 0
            for i in range(N_NODES):
                p = f"{TAG}__{BN}__{i}"
                th = z[f"{p}__{grp}_proto"].astype(np.float64)
                zp = z[f"{p}__{grp}_zperp"].astype(np.float64)
                c = 2.0 * np.sin(np.deg2rad(th) / 2.0)
                ch.append(c)
                rs.append(zp)
                bad += int((zp > c + 1e-6).sum())
            c, r = np.concatenate(ch), np.concatenate(rs)
            print(f"  {SHORT[d]:<9}{lab:<10}{c.mean():>11.4f}{r.mean():>11.4f}"
                  f"{(c.mean()-r.mean()):>9.4f}{bad:>8}")

    print("\n【4】為什麼排序會變好：均值差不夠看，要看「相對於散佈」的分離度")
    print("     d = (未知均值 − 已知均值) / 合併標準差；AUROC 對單調變換不變 ⇒ 弦距與角距離同值")
    from sklearn.metrics import roc_auc_score
    print(f"  {'fold':<9}{'點 d':>9}{'面 d':>9}{'Δd':>9}   {'點 AUROC':>10}{'面 AUROC':>10}{'Δ':>9}")
    for d in PACS:
        z = np.load(f"{NPZ_DIR}/{d}.npz", allow_pickle=True)
        row = {}
        for ro, conv in [("proto", lambda v: 2.0 * np.sin(np.deg2rad(v) / 2.0)),
                         ("zperp", lambda v: v)]:
            kk, uu = [], []
            for i in range(N_NODES):
                pp = f"{TAG}__{BN}__{i}"
                kk.append(conv(z[f"{pp}__tgt_known_{ro}"].astype(np.float64)))
                uu.append(conv(z[f"{pp}__tgt_unk_{ro}"].astype(np.float64)))
            k, u = np.concatenate(kk), np.concatenate(uu)
            sd = np.sqrt((k.var() * len(k) + u.var() * len(u)) / (len(k) + len(u)))
            auc = roc_auc_score(np.r_[np.zeros(len(k)), np.ones(len(u))], np.r_[k, u])
            row[ro] = ((u.mean() - k.mean()) / sd, auc)
        print(f"  {SHORT[d]:<9}{row['proto'][0]:>9.3f}{row['zperp'][0]:>9.3f}"
              f"{row['zperp'][0]-row['proto'][0]:>+9.3f}   "
              f"{row['proto'][1]:>10.4f}{row['zperp'][1]:>10.4f}{row['zperp'][1]-row['proto'][1]:>+9.4f}")


if __name__ == "__main__":
    main()
