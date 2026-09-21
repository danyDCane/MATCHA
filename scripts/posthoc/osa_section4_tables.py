"""0905 §4「操作點各指標」四張表的完整版（含面讀出 `‖z⊥‖`）。

背景：§4 原本只有 energy 與原型角距離（點）。2026-09-10 dany 裁定採用**面讀出**、
且對外比法定為【我方+面／平均B vs StyleDDG+energy／原樣】⇒ §4 四張表必須含面讀出，
否則主組合的三個門檻相關指標在 §4 查不到（dany 2026-09-10 指出）。

輸出四張表：
  4.1 closed_acc  ——【與讀出無關】同一顆 checkpoint 的六類頭 top-1（換讀出不會變）
  4.2 誤拒率      —— 已知類被拒絕的比例（設計值 0.05）
  4.3 放行率      —— 未知類被誤放行的比例（＝1−正確拒絕率）
  4.4 部署 AUROC  —— 門檻無關

用法：./venv_matcha/bin/python scripts/posthoc/osa_section4_tables.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score

PACS = ["art_painting", "cartoon", "photo", "sketch"]
SH = {"art_painting": "art", "cartoon": "cartoon", "photo": "photo", "sketch": "sketch"}
N, Q, PI = 9, 0.95, 0.2
ROWS = [("StyleDDG+energy／原樣", "baseline", "raw", "energy"),
        ("StyleDDG+energy／平均B", "baseline", "avg", "energy"),
        ("我方+原型(點)／原樣", "ours", "raw", "proto"),
        ("我方+原型(點)／平均B", "ours", "avg", "proto"),
        ("我方+‖z⊥‖(面)／原樣", "ours", "raw", "zperp"),
        ("我方+‖z⊥‖(面)／平均B", "ours", "avg", "zperp"),
        ("我方+energy／原樣", "ours", "raw", "energy"),
        ("我方+energy／平均B", "ours", "avg", "energy")]


def metrics(fold, tag, bn, ro):
    z = np.load(f"results/osa/{fold}.npz", allow_pickle=True)
    if f"{tag}__{bn}__0__src_{ro}" not in z:
        return None
    fpr, pas, au, acc = [], [], [], []
    for i in range(N):
        p = f"{tag}__{bn}__{i}"
        s = z[f"{p}__src_{ro}"].astype(np.float64)
        k = z[f"{p}__tgt_known_{ro}"].astype(np.float64)
        u = z[f"{p}__tgt_unk_{ro}"].astype(np.float64)
        c = z[f"{p}__tgt_known_correct"] > 0
        tau = np.quantile(s, Q)
        fpr.append(float((k > tau).mean()))        # 誤拒率
        pas.append(float((u <= tau).mean()))       # 放行率
        au.append(roc_auc_score(np.r_[np.zeros(len(k)), np.ones(len(u))], np.r_[k, u]))
        acc.append(float(c.mean()) * 100)          # closed_acc（與讀出無關）
    return tuple(float(np.mean(x)) for x in (acc, fpr, pas, au))


def main():
    R = {n: {f: metrics(f, t, b, r) for f in PACS} for n, t, b, r in ROWS}
    for idx, (title, note, fmt) in enumerate([
            ("4.1 closed_acc（%，泛化，無檢測參與）", "⚠️ 與讀出無關：同一顆 checkpoint 換讀出不會變", "{:.2f}"),
            ("4.2 誤拒率（設計值 0.05，越低越好）", "已知類被拒絕的比例", "{:.4f}"),
            ("4.3 放行率（＝1−正確拒絕率，越低越好）", "未知類被誤放行的比例", "{:.4f}"),
            ("4.4 部署 AUROC（門檻無關，越高越好）", "只量排序，與門檻放哪無關", "{:.4f}")]):
        print("\n" + "=" * 92)
        print(f"§{title}    {note}")
        print("=" * 92)
        print(f"  {'組合':<24}" + "".join(f"{SH[f]:>10}" for f in PACS) + f"{'平均':>10}")
        seen = set()
        for n, *_ in ROWS:
            v = [R[n][f] for f in PACS]
            if any(x is None for x in v):
                continue
            col = [x[idx] for x in v]
            if idx == 0:                       # closed_acc：讀出無關 ⇒ 只印去重後的臂
                key = (n.split("＋")[0].split("+")[0], n.split("／")[-1])
                if key in seen:
                    continue
                seen.add(key)
                n = n.split("+")[0] + "／" + n.split("／")[-1]
            print(f"  {n:<24}" + "".join(fmt.format(x).rjust(10) for x in col)
                  + fmt.format(np.mean(col)).rjust(10))

    print("\n" + "=" * 92)
    print("★ 主組合（2026-09-10 起的對外比法）：我方+‖z⊥‖(面)／平均B  vs  StyleDDG+energy／原樣")
    print("=" * 92)
    o, b = R["我方+‖z⊥‖(面)／平均B"], R["StyleDDG+energy／原樣"]
    for idx, (nm, fmt, hi) in enumerate([("closed_acc(%)", "{:+.2f}", True), ("誤拒率", "{:+.4f}", False),
                                         ("放行率", "{:+.4f}", False), ("部署AUROC", "{:+.4f}", True)]):
        d = [o[f][idx] - b[f][idx] for f in PACS]
        win = sum(1 for x in d if (x > 0) == hi)
        print(f"  {nm:<14}" + "".join(fmt.format(x).rjust(10) for x in d)
              + fmt.format(np.mean(d)).rjust(10) + f"   我方較優 {win}/4")
    print("\n  （誤拒率／放行率越低越好 ⇒ 負號＝我方較優）")


def osa_calibration_table():
    """§4.7｜OSA 的四種口徑比法（原 §8，2026-09-14 併入並補上面讀出）。

    ⚠️ 原 §8 只有點讀出，而面讀出已於 2026-09-10 裁定採用 ⇒ 該表過期，本函式取代之。
    """
    def osa(fold, tag, bn, ro):
        z = np.load(f"results/osa/{fold}.npz", allow_pickle=True)
        if f"{tag}__{bn}__0__src_{ro}" not in z:
            return None
        v = []
        for i in range(N):
            p = f"{tag}__{bn}__{i}"
            s_ = z[f"{p}__src_{ro}"].astype(np.float64)
            k = z[f"{p}__tgt_known_{ro}"].astype(np.float64)
            u = z[f"{p}__tgt_unk_{ro}"].astype(np.float64)
            c = z[f"{p}__tgt_known_correct"] > 0
            t = np.quantile(s_, Q)
            v.append(100 * ((1 - PI) * float(((k <= t) & c).mean()) + PI * float((u > t).mean())))
        return float(np.mean(v))

    W = 96
    print("\n" + "=" * W)
    print("★ OSA 的四種口徑比法（π=20%；正號＝我方較優）")
    print("=" * W)
    CMP = [("① 同口徑・都不匯聚 BN", ("ours", "raw"), ("baseline", "raw")),
           ("② 同口徑・都匯聚 BN", ("ours", "avg"), ("baseline", "avg")),
           ("③ 現行對外比法（我方平均B vs SOTA 原樣）", ("ours", "avg"), ("baseline", "raw"))]
    print(f"\n  {'比法':<36}{'我方讀出':<14}" + "".join(f"{SH[f]:>9}" for f in PACS)
          + f"{'平均Δ':>9}{'勝場':>7}")
    for lbl, (oa, ob), (ba, bb_) in CMP:
        for j, ro in enumerate(["proto", "zperp"]):
            d = [osa(f, oa, ob, ro) - osa(f, ba, bb_, "energy") for f in PACS]
            nm = "原型角距離(點)" if ro == "proto" else "‖z⊥‖殘差(面)"
            print(f"  {lbl if j == 0 else '':<36}{nm:<14}" + "".join(f"{x:>+9.2f}" for x in d)
                  + f"{np.mean(d):>+9.2f}{sum(1 for x in d if x > 0):>5}/4")
    d = [osa(f, "ours", "raw", "energy") - osa(f, "baseline", "raw", "energy") for f in PACS]
    print(f"  {'④ 單一變因＝模型（雙方都用 energy・原樣）':<36}{'energy':<14}"
          + "".join(f"{x:>+9.2f}" for x in d) + f"{np.mean(d):>+9.2f}{sum(1 for x in d if x>0):>5}/4")
    print("\n  ⚠️ ③ 自 2026-09-10 起為現行對外比法（BN 匯聚已裁定為我方方法元件）；")
    print("     ② 為必附的 ablation（若 baseline 也拿到該元件會如何）。")


if __name__ == "__main__":
    main()
    osa_calibration_table()
