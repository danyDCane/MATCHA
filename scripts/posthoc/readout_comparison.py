"""六個讀出的完整對照：哪一個讀出在部署上最好？（dany 2026-09-09）

要回答的問題：面讀出 ‖z⊥‖ 讓部署 AUROC 變好，OSA 會不會跟著變好？
⇒ 拆成兩個來源：① 排序變好（AUROC 反映）② 門檻位置變對（AUROC 看不見，Q1 分位漂移反映）。

六個讀出（全部定向為 高＝越像 OOD）：
  energy   = −logsumexp(logits)          ← 文獻標準分數
  msp      = −max softmax(logits)        ← 最常見的 baseline
  maxlogit = −max(logits)
  negstd   = −std(logits)                ← TaskBoard 記為「同模型最強免費讀出」，必須誠實同報
  proto    = min_c arccos(z·c_c)（度）    ← 我方原型角距離（點）
  zperp    = ‖z⊥‖ 對六類別中心子空間的殘差 ← 我方面讀出（面）

⚠️ zperp/proto 只在有 prototypes buffer 的 checkpoint 上存在（baseline 無）。
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import roc_auc_score

PACS = ["art_painting", "cartoon", "photo", "sketch"]
SH = {"art_painting": "art", "cartoon": "cartoon", "photo": "photo", "sketch": "sketch"}
N, PI, Q = 9, 0.2, 0.95
RO = ["energy", "msp", "maxlogit", "negstd", "proto", "zperp"]
RN = {"energy": "energy", "msp": "MSP", "maxlogit": "MaxLogit", "negstd": "−std(logit)",
      "proto": "原型角距離(點)", "zperp": "‖z⊥‖殘差(面)"}


def stats(fold, tag, bn, ro):
    z = np.load(f"results/osa/{fold}.npz", allow_pickle=True)
    if f"{tag}__{bn}__0__src_{ro}" not in z:
        return None
    acc = {k: [] for k in ["auroc", "osa", "fpr", "r_ood", "a_id", "shift", "decis", "fuzzy"]}
    for i in range(N):
        p = f"{tag}__{bn}__{i}"
        s_src, s_k, s_u = (z[f"{p}__src_{ro}"].astype(np.float64),
                           z[f"{p}__tgt_known_{ro}"].astype(np.float64),
                           z[f"{p}__tgt_unk_{ro}"].astype(np.float64))
        corr = z[f"{p}__tgt_known_correct"] > 0
        tau, sig = float(np.quantile(s_src, Q)), s_src.std()
        acc["auroc"].append(roc_auc_score(np.r_[np.zeros(len(s_k)), np.ones(len(s_u))], np.r_[s_k, s_u]))
        a_id = float(((s_k <= tau) & corr).mean()); r_ood = float((s_u > tau).mean())
        acc["a_id"].append(a_id); acc["r_ood"].append(r_ood)
        acc["osa"].append(((1 - PI) * a_id + PI * r_ood) * 100)
        acc["fpr"].append(float((s_k > tau).mean()))
        acc["shift"].append(float((s_src < np.median(s_k)).mean()) * 100)
        d = np.r_[np.abs(s_k - tau), np.abs(s_u - tau)] / sig
        w = np.r_[np.full(len(s_k), (1 - PI) / len(s_k)), np.full(len(s_u), PI / len(s_u))]
        o = np.argsort(d); cw = np.cumsum(w[o]) / w.sum()
        acc["decis"].append(d[o][np.searchsorted(cw, 0.5)])
        acc["fuzzy"].append(float((np.abs(s_k - tau) / sig < 0.5).mean()) * (1 - PI)
                            + float((np.abs(s_u - tau) / sig < 0.5).mean()) * PI)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    W = 104
    ARMS = [("baseline", "raw", "StyleDDG/原樣"), ("baseline", "avg", "StyleDDG/平均B"),
            ("ours", "raw", "我方/原樣"), ("ours", "avg", "我方/平均B")]
    R = {}
    for tag, bn, _ in ARMS:
        for ro in RO:
            for f in PACS:
                s = stats(f, tag, bn, ro)
                if s: R[(tag, bn, ro, f)] = s
    def m(tag, bn, ro, key):
        v = [R[(tag, bn, ro, f)][key] for f in PACS if (tag, bn, ro, f) in R]
        return float(np.mean(v)) if len(v) == len(PACS) else None

    for tag, bn, arm in ARMS:
        print("\n" + "=" * W); print(f"★ {arm}（四 fold 平均）"); print("=" * W)
        print(f"  {'讀出':<16}{'部署AUROC':>11}{'OSA':>8}{'誤拒率':>9}{'正確拒絕':>10}"
              f"{'分位漂移':>10}{'果斷度':>9}{'模糊帶':>9}")
        rows = []
        for ro in RO:
            a = m(tag, bn, ro, "auroc")
            if a is None: continue
            rows.append((ro, a, m(tag,bn,ro,"osa"), m(tag,bn,ro,"fpr"), m(tag,bn,ro,"r_ood"),
                         m(tag,bn,ro,"shift"), m(tag,bn,ro,"decis"), m(tag,bn,ro,"fuzzy")))
        for r in rows:
            print(f"  {RN[r[0]]:<16}{r[1]:>11.4f}{r[2]:>8.2f}{r[3]:>9.4f}{r[4]:>10.4f}"
                  f"{r[5]:>9.1f}%{r[6]:>9.3f}{r[7]:>9.3f}")
        ba = max(rows, key=lambda r: r[1]); bo = max(rows, key=lambda r: r[2])
        print(f"  ⇒ AUROC 最佳：**{RN[ba[0]]}** ({ba[1]:.4f})　｜　OSA 最佳：**{RN[bo[0]]}** ({bo[2]:.2f})"
              + ("　★ 兩者不同！" if ba[0] != bo[0] else "　（一致）"))

    print("\n" + "=" * W); print("★★ 逐 fold 的部署 AUROC 與 OSA（我方/平均B，看 sketch 破口）"); print("=" * W)
    for key, lab in [("auroc", "部署 AUROC"), ("osa", "OSA(π=20%)")]:
        print(f"\n  {lab}")
        print(f"    {'讀出':<16}" + "".join(f"{SH[f]:>10}" for f in PACS) + f"{'平均':>10}")
        for ro in RO:
            v = [R[("ours","avg",ro,f)][key] for f in PACS if ("ours","avg",ro,f) in R]
            if len(v) == len(PACS):
                fmt = "{:>10.4f}" if key == "auroc" else "{:>10.2f}"
                print(f"    {RN[ro]:<16}" + "".join(fmt.format(x) for x in v) + fmt.format(np.mean(v)))

    print("\n" + "=" * W); print("★★★ 核心問題：AUROC 的提升有多少轉成 OSA？（我方/平均B，以角距離為基準）"); print("=" * W)
    base_a, base_o = m("ours","avg","proto","auroc"), m("ours","avg","proto","osa")
    print(f"  基準＝原型角距離：AUROC {base_a:.4f}、OSA {base_o:.2f}、分位漂移 {m('ours','avg','proto','shift'):.1f}%")
    print(f"\n  {'讀出':<16}{'ΔAUROC':>10}{'ΔOSA(pp)':>11}{'轉換率':>10}{'Δ分位漂移':>12}{'Δ誤拒率':>10}")
    for ro in RO:
        if ro == "proto" or m("ours","avg",ro,"auroc") is None: continue
        da = m("ours","avg",ro,"auroc") - base_a
        do = m("ours","avg",ro,"osa") - base_o
        conv = f"{do/(da*100):>9.2f}" if abs(da) > 1e-6 else "       n/a"
        print(f"  {RN[ro]:<16}{da:>+10.4f}{do:>+11.2f}{conv}"
              f"{m('ours','avg',ro,'shift')-m('ours','avg','proto','shift'):>+11.1f}%"
              f"{m('ours','avg',ro,'fpr')-m('ours','avg','proto','fpr'):>+10.4f}")
    print("\n  轉換率 ＝ ΔOSA(pp) ÷ (ΔAUROC×100)：>1 表示 AUROC 的提升被放大成部署效用；<1 表示打折。")

    # ── 逐張去向表：固定「我方/平均B」，只換讀出 ──
    NK, NU = int(1000*(1-PI)), int(1000*PI)
    print("\n" + "=" * W)
    print(f"★★ 逐張去向（假想 1000 張，π={PI:.0%} ⇒ 已知 {NK}、未知 {NU}）｜固定 我方/平均B，只換讀出")
    print("=" * W)
    hdr = (f"    {'讀出':<16}{'已知:誤拒':>10}{'放行分錯':>10}{'放行分對✓':>11}"
           f"{'未知:拒絕✓':>12}{'放行':>7}{'總對':>8}{'acc':>9}")
    def cells(d):
        rej, ok_k = d["fpr"]*NK, d["a_id"]*NK
        ok_u = d["r_ood"]*NU
        return rej, NK-rej-ok_k, ok_k, ok_u, NU-ok_u, ok_k+ok_u
    for f in PACS + ["平均"]:
        print(f"\n  ── {SH.get(f, f)} ──"); print(hdr)
        # 參照：無檢測（我方分類器，全部放行、未知全錯）
        if f == "平均":
            A = float(np.mean([R[("ours","avg","energy",x)]["a_id"] +
                               R[("ours","avg","energy",x)]["fpr"]*0 for x in PACS]))
        base = {}
        for ro in ["energy","msp","maxlogit","negstd","proto","zperp"]:
            if f == "平均":
                d = {k: float(np.mean([R[("ours","avg",ro,x)][k] for x in PACS]))
                     for k in ["a_id","r_ood","fpr"]} if ("ours","avg",ro,PACS[0]) in R else None
            else:
                d = R.get(("ours","avg",ro,f))
            if not d: continue
            c = cells(d)
            star = " ★" if ro == "zperp" else ""
            print(f"    {RN[ro]:<16}{c[0]:>10.0f}{c[1]:>10.0f}{c[2]:>11.0f}"
                  f"{c[3]:>12.0f}{c[4]:>7.0f}{c[5]:>8.0f}{c[5]/10:>8.1f}%{star}")


if __name__ == "__main__":
    main()
