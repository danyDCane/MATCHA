"""為什麼 AUROC 較低的讀出反而 OSA 較高？——把「排序」與「門檻」拆開（dany 2026-09-09）

起因：dany 問「面讀出在 sketch 的部署 AUROC 輸 energy，為什麼 OSA 反而贏？」

拆法（兩張表，互相印證）：

【表一】用「作弊門檻」把兩件事分離
  OSA@oracle = 掃過所有可能門檻取 OSA 最大值（需要知道未知類答案才選得到，僅作上限）
             ⇒ 這一欄只反映「排序好不好」，跟 AUROC 同向
  OSA@src95  = 實際用的門檻（來源域已知類分數的 95 分位，逐節點）
  門檻代價    = OSA@src95 − OSA@oracle（≤0）
             ⇒ 這一欄只反映「門檻位置對不對」，AUROC 完全看不見
  恆等式：OSA@src95 = OSA@oracle + 門檻代價
         ⇒ 兩讀出的 OSA 勝差 = 排序差 + 門檻代價差

【表二】門檻代價從哪來（同樣以來源域標準差 σ_src 為尺）
  A 門檻頭距 = (τ − median(來源域已知)) / σ_src      門檻離來源域中位數多遠
  B 目標漂移 = (median(目標域已知) − median(來源域已知)) / σ_src   換畫風後整體漂多遠
  剩餘餘裕  = A − B                                  漂完之後門檻還剩多少緩衝
             ⇒ 若剩餘餘裕與誤拒率單調反向，即證實「門檻代價 = 餘裕被漂移吃光」

輸入：results/osa/{fold}.npz（由 scripts/open_set_accuracy.py infer 產生）
用法：./venv_matcha/bin/python scripts/posthoc/auroc_vs_osa_decomposition.py
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
TAG, BN = "ours", "avg"          # 表一～三固定 我方/BN 平均B，只換讀出（與 §7.4 同口徑）


def decompose(fold, ro):
    """回傳該 fold 該讀出的 9 節點平均；讀出不存在時回 None。"""
    z = np.load(f"results/osa/{fold}.npz", allow_pickle=True)
    if f"{TAG}__{BN}__0__src_{ro}" not in z:
        return None
    acc = {k: [] for k in ["auroc", "osa", "oracle", "cost", "A", "B", "slack", "fpr", "r_ood"]}
    for i in range(N):
        p = f"{TAG}__{BN}__{i}"
        s_src = z[f"{p}__src_{ro}"].astype(np.float64)
        s_k = z[f"{p}__tgt_known_{ro}"].astype(np.float64)
        s_u = z[f"{p}__tgt_unk_{ro}"].astype(np.float64)
        corr = z[f"{p}__tgt_known_correct"] > 0
        n_k, n_u = len(s_k), len(s_u)
        tau, sig, med_s = float(np.quantile(s_src, Q)), s_src.std(), float(np.median(s_src))

        acc["auroc"].append(roc_auc_score(np.r_[np.zeros(n_k), np.ones(n_u)], np.r_[s_k, s_u]))

        # 實際門檻（來源域 95 分位）：放行且分對 / 未知被拒
        a_id = float(((s_k <= tau) & corr).mean())
        r_ood = float((s_u > tau).mean())
        acc["osa"].append(100 * ((1 - PI) * a_id + PI * r_ood))
        acc["fpr"].append(float((s_k > tau).mean()))
        acc["r_ood"].append(r_ood)

        # 作弊門檻：掃過所有候選門檻取最大 OSA（向量化，非逐點迴圈）
        cand = np.r_[-np.inf, np.unique(np.r_[s_k, s_u])]
        a_c = np.searchsorted(np.sort(s_k[corr]), cand, side="right") / n_k
        r_c = (n_u - np.searchsorted(np.sort(s_u), cand, side="right")) / n_u
        oracle = 100 * float(((1 - PI) * a_c + PI * r_c).max())
        acc["oracle"].append(oracle)
        acc["cost"].append(100 * ((1 - PI) * a_id + PI * r_ood) - oracle)

        # 門檻頭距 / 目標漂移 / 剩餘餘裕（單位：σ_src）
        A = (tau - med_s) / sig
        B = (float(np.median(s_k)) - med_s) / sig
        acc["A"].append(A); acc["B"].append(B); acc["slack"].append(A - B)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    W = 100
    R = {f: {ro: decompose(f, ro) for ro in RO} for f in PACS}

    print("=" * W)
    print(f"★ 表一｜排序 vs 門檻的分離（{TAG}/{BN}＝我方/BN 平均B，9 節點平均，π={PI:.0%}）")
    print("   恆等式：OSA@src95 ＝ OSA@oracle ＋ 門檻代價")
    print("=" * W)
    for f in PACS:
        print(f"\n  ── {SH[f]} ──")
        print(f"    {'讀出':<18}{'部署AUROC':>10}{'OSA@oracle':>12}{'門檻代價':>10}{'OSA@src95':>11}")
        for ro in RO:
            v = R[f][ro]
            if v is None:
                continue
            print(f"    {RN[ro]:<18}{v['auroc']:>10.4f}{v['oracle']:>12.2f}"
                  f"{v['cost']:>10.2f}{v['osa']:>11.2f}")

    print("\n" + "=" * W)
    print(f"★ 表二｜門檻代價從哪來（單位：來源域標準差 σ_src）")
    print("   剩餘餘裕 ＝ A 門檻頭距 − B 目標漂移；若與誤拒率單調反向即證實機制")
    print("=" * W)
    for f in PACS:
        print(f"\n  ── {SH[f]} ──")
        print(f"    {'讀出':<18}{'A:門檻頭距':>11}{'B:目標漂移':>11}{'剩餘餘裕':>10}"
              f"{'誤拒率':>9}{'正確拒絕':>9}")
        for ro in RO:
            v = R[f][ro]
            if v is None:
                continue
            print(f"    {RN[ro]:<18}{v['A']:>11.2f}{v['B']:>11.2f}{v['slack']:>10.2f}"
                  f"{v['fpr']:>9.4f}{v['r_ood']:>9.4f}")

    print("\n" + "=" * W)
    print("★ 表三｜面讀出 vs energy 的逐 fold 對帳（正號＝面讀出較優）")
    print("=" * W)
    print(f"    {'fold':<10}{'ΔAUROC':>10}{'Δ排序(oracle)':>15}{'Δ門檻代價':>12}{'ΔOSA':>9}{'對帳':>9}")
    for f in PACS:
        e, zp = R[f]["energy"], R[f]["zperp"]
        d_rank = zp["oracle"] - e["oracle"]
        d_cost = zp["cost"] - e["cost"]
        d_osa = zp["osa"] - e["osa"]
        print(f"    {SH[f]:<10}{zp['auroc']-e['auroc']:>+10.4f}{d_rank:>+15.2f}"
              f"{d_cost:>+12.2f}{d_osa:>+9.2f}{d_rank+d_cost-d_osa:>9.2e}")
    ag = {k: [] for k in ["auroc", "rank", "cost", "osa"]}
    for f in PACS:
        e, zp = R[f]["energy"], R[f]["zperp"]
        ag["auroc"].append(zp["auroc"] - e["auroc"]); ag["rank"].append(zp["oracle"] - e["oracle"])
        ag["cost"].append(zp["cost"] - e["cost"]); ag["osa"].append(zp["osa"] - e["osa"])
    m = {k: float(np.mean(v)) for k, v in ag.items()}
    print(f"    {'平均':<8}{m['auroc']:>+10.4f}{m['rank']:>+15.2f}{m['cost']:>+12.2f}{m['osa']:>+9.2f}")
    print("\n  （最後一欄＝排序差＋門檻代價差−ΔOSA，應為 0，用來驗證分解無殘差）")
    print(f"  （平均列自檢：排序 {m['rank']:+.2f} ＋ 門檻 {m['cost']:+.2f} ＝ {m['rank']+m['cost']:+.2f}"
          f" ＝ ΔOSA {m['osa']:+.2f}；並應等於 §7.4 的 68.22−67.61）")


def cost_of(fold, tag, bn, ro):
    """單一組合的門檻代價（9 節點平均，負值＝比事後最佳門檻差多少 pp）。"""
    global TAG, BN
    _t, _b = TAG, BN
    TAG, BN = tag, bn
    try:
        v = decompose(fold, ro)
    finally:
        TAG, BN = _t, _b
    return None if v is None else v["cost"]


def baseline_comparisons():
    """§7.5.5｜門檻代價的四種比法（含 baseline 兩臂）——全部由判決數構成、無單位選擇。"""
    W = 96
    A = {"SOTA/原樣": ("baseline", "raw", "energy"), "SOTA/平均B": ("baseline", "avg", "energy"),
         "我方energy/原樣": ("ours", "raw", "energy"), "我方energy/平均B": ("ours", "avg", "energy"),
         "我方點/原樣": ("ours", "raw", "proto"), "我方點/平均B": ("ours", "avg", "proto"),
         "我方面/原樣": ("ours", "raw", "zperp"), "我方面/平均B": ("ours", "avg", "zperp")}
    C = {n: [cost_of(f, *a) for f in PACS] for n, a in A.items()}
    print("\n" + "=" * W)
    print("★ 門檻代價（來源域95分位門檻的 OSA − 事後最佳門檻的 OSA，pp；越接近 0 越好）")
    print("=" * W)
    print(f"\n  {'組合':<20}" + "".join(f"{SH[f]:>9}" for f in PACS) + f"{'平均':>9}")
    for n, v in C.items():
        print(f"  {n:<20}" + "".join(f"{x:>9.2f}" for x in v) + f"{np.mean(v):>9.2f}")
    print("\n" + "=" * W)
    print("★ 四種比法：我方少付多少門檻代價（正號＝我方較好）")
    print("=" * W)
    CMP = [("① 同模型換讀出（點 vs 自家energy・平均B）", "我方點/平均B", "我方energy/平均B"),
           ("② 對外靶・同口徑都不匯聚BN（點 vs SOTA）", "我方點/原樣", "SOTA/原樣"),
           ("③ 對外靶・同口徑都匯聚BN（點 vs SOTA）", "我方點/平均B", "SOTA/平均B"),
           ("④ 現行對外比法（我方面/平均B vs SOTA/原樣）", "我方面/平均B", "SOTA/原樣"),
           ("⑤ 面 vs 自家energy（平均B）", "我方面/平均B", "我方energy/平均B"),
           ("⑥ 面・同口徑都匯聚BN", "我方面/平均B", "SOTA/平均B")]
    print(f"\n  {'比法':<42}" + "".join(f"{SH[f]:>9}" for f in PACS) + f"{'平均':>9}{'勝場':>7}")
    for lbl, o, b in CMP:
        d = [abs(y) - abs(x) for x, y in zip(C[o], C[b])]
        print(f"  {lbl:<42}" + "".join(f"{x:>+9.2f}" for x in d)
              + f"{np.mean(d):>+9.2f}{sum(1 for x in d if x > 0):>5}/4")
    print("\n  ⚠️ ④ 是 2026-09-10 起的對外比法（BN 匯聚已定位為我方方法元件）；⑥ 為必附的 ablation。")


if __name__ == "__main__":
    main()
    baseline_comparisons()
