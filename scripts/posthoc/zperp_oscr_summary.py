"""面讀出 `‖z⊥‖` 進 OSCR：四 fold × {BN 原樣, 平均B} × {baseline, 我方} 彙總（dany 2026-09-09）

要回答的問題：我們的 OSCR 目前用「點」讀出（`proto_angle`），換成「面」讀出（`zperp`）會怎樣？

⚠️ 三個必須一起看的口徑：
  ① **對外靶＝StyleDDG baseline · energy · BN 原樣 · 四折**（TaskBoard §A，2026-09-09 dany 裁定）
  ② **內部門檻＝我們自己模型上的 energy**——誠實對照，不是判決依據
  ③ **點 vs 面是同一顆 checkpoint、同一次前向**⇒ closed_acc 完全相同
     ⇒ OSCR 的差**純粹來自檢測**（不像跟 baseline 比時混了泛化）

輸入：research/V1_baseline/0909_zperp_oscr_4fold.csv
用法：./venv_matcha/bin/python scripts/posthoc/zperp_oscr_summary.py
"""
import csv, statistics, collections

CSV = "research/V1_baseline/0909_zperp_oscr_4fold.csv"
PACS = ["art_painting", "cartoon", "photo", "sketch"]
SH = {"art_painting": "art", "cartoon": "cartoon", "photo": "photo", "sketch": "sketch"}
RN = {"msp": "MSP", "energy": "energy", "proto_angle": "原型角距離(點)", "zperp": "‖z⊥‖殘差(面)"}


def load():
    rows = [r for r in csv.DictReader(open(CSV)) if r["ckpt"] == "final"]
    out = collections.defaultdict(dict)          # (arm, bn, fold) -> {score_fn: {metric: mean}}
    for r in rows:
        arm = "ours" if "p1a_async" in r["run"] else "baseline"
        bn = "avg" if r["topo"].endswith("avgbn") else "raw"
        out[(arm, bn, r["leave_out"])].setdefault(r["score_fn"], []).append(r)
    agg = {}
    for k, d in out.items():
        agg[k] = {sf: {m: statistics.mean(float(x[m]) for x in v)
                       for m in ["oscr", "det_auroc", "closed_acc"]}
                  for sf, v in d.items()}
    return agg


def table(agg, arm, bn, metric, title):
    print(f"\n  {title}")
    print(f"    {'讀出':<18}" + "".join(f"{SH[f]:>10}" for f in PACS) + f"{'平均':>10}{'勝場*':>8}")
    base = {f: agg[("baseline", "raw", f)]["energy"][metric] for f in PACS}   # 對外靶
    for sf in ["msp", "energy", "proto_angle", "zperp"]:
        vals = [agg[(arm, bn, f)].get(sf, {}).get(metric) for f in PACS]
        if any(v is None for v in vals):
            continue
        win = sum(1 for f, v in zip(PACS, vals) if v > base[f])
        print(f"    {RN[sf]:<18}" + "".join(f"{v:>10.4f}" for v in vals)
              + f"{statistics.mean(vals):>10.4f}{win:>6}/4")


def main():
    agg = load()
    W = 92
    print("=" * W)
    print("★ 對外靶（TaskBoard §A）＝ StyleDDG baseline · energy · BN 原樣 · 四折")
    print("=" * W)
    for m, t in [("oscr", "OSCR"), ("det_auroc", "部署 AUROC"), ("closed_acc", "closed_acc")]:
        v = [agg[("baseline", "raw", f)]["energy"][m] for f in PACS]
        print(f"  {t:<12}" + "".join(f"{x:>10.4f}" for x in v) + f"{statistics.mean(v):>10.4f}")
    print(f"  {'(fold)':<12}" + "".join(f"{SH[f]:>10}" for f in PACS))

    for arm, bn, title in [("ours", "raw", "我方 · BN 原樣"), ("ours", "avg", "我方 · BN 平均B"),
                           ("baseline", "avg", "StyleDDG · BN 平均B")]:
        print("\n" + "=" * W)
        print(f"★ {title}")
        print("=" * W)
        for m, t in [("oscr", "OSCR"), ("det_auroc", "部署 AUROC"), ("closed_acc", "closed_acc")]:
            table(agg, arm, bn, m, t)
    print("\n  *勝場＝該 fold 贏過【對外靶】（baseline·energy·BN 原樣）的折數")

    print("\n" + "=" * W)
    print("★★ 核心：點 → 面（同一 checkpoint、同一次前向 ⇒ closed_acc 相同 ⇒ 差純粹是檢測）")
    print("=" * W)
    for bn, t in [("raw", "BN 原樣"), ("avg", "BN 平均B")]:
        print(f"\n  ── 我方 · {t} ──")
        print(f"    {'量':<14}" + "".join(f"{SH[f]:>10}" for f in PACS) + f"{'平均':>10}")
        for m, lbl in [("oscr", "OSCR"), ("det_auroc", "部署AUROC")]:
            p = [agg[("ours", bn, f)]["proto_angle"][m] for f in PACS]
            z = [agg[("ours", bn, f)]["zperp"][m] for f in PACS]
            e = [agg[("ours", bn, f)]["energy"][m] for f in PACS]
            print(f"    {lbl+'·點':<14}" + "".join(f"{x:>10.4f}" for x in p) + f"{statistics.mean(p):>10.4f}")
            print(f"    {lbl+'·面':<14}" + "".join(f"{x:>10.4f}" for x in z) + f"{statistics.mean(z):>10.4f}")
            d = [b - a for a, b in zip(p, z)]
            print(f"    {lbl+' Δ(面−點)':<14}" + "".join(f"{x:>+10.4f}" for x in d)
                  + f"{statistics.mean(d):>+10.4f}   同號{sum(1 for x in d if x>0)}/4")
            print(f"    {lbl+'·energy':<14}" + "".join(f"{x:>10.4f}" for x in e)
                  + f"{statistics.mean(e):>10.4f}   ← 內部門檻")
            de = [b - a for a, b in zip(e, z)]
            print(f"    {lbl+' Δ(面−en)':<14}" + "".join(f"{x:>+10.4f}" for x in de)
                  + f"{statistics.mean(de):>+10.4f}   勝{sum(1 for x in de if x>0)}/4")
            print()
    print("  ⚠️ OSCR 上限 ＝ closed_acc（y 軸只算「分類正確 AND 被接受」）⇒ 兩者不可分開解讀。")



def compare_to_baseline(agg):
    """對 StyleDDG 的三種比法（沿用 0905 §8 的口徑框）。"""
    W = 92
    print("\n" + "=" * W)
    print("★★★ 對 StyleDDG 的三種比法（勝場＝逐折贏的折數）")
    print("=" * W)
    COMBOS = [
        ("(1) 同口徑,都不做BN匯聚", ("ours", "raw"), ("baseline", "raw")),
        ("(2) 同口徑,都做BN匯聚", ("ours", "avg"), ("baseline", "avg")),
        ("(3) !!不對等(我方平均B vs SOTA原樣)=TaskBoard §A現行", ("ours", "avg"), ("baseline", "raw")),
    ]
    for m, lbl in [("oscr", "OSCR"), ("det_auroc", "部署 AUROC")]:
        print(f"\n  -- {lbl} --")
        print(f"    {'比法':<48}{'我方讀出':<15}" + "".join(f"{SH[f]:>9}" for f in PACS)
              + f"{'平均':>9}{'勝場':>7}")
        for title, (oa, ob), (ba, bb_) in COMBOS:
            for sf in ["proto_angle", "zperp"]:
                d = [agg[(oa, ob, f)][sf][m] - agg[(ba, bb_, f)]["energy"][m] for f in PACS]
                print(f"    {title if sf == 'proto_angle' else '':<48}{RN[sf]:<15}"
                      + "".join(f"{x:>+9.4f}" for x in d)
                      + f"{statistics.mean(d):>+9.4f}{sum(1 for x in d if x > 0):>5}/4")
    print("\n  !! (3) 兩邊 BN 口徑不同、不是可辯護的比較；TaskBoard §A 的 +0.0263／+0.0222 走的是這條。")


if __name__ == "__main__":
    main()
    compare_to_baseline(load())
