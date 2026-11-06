"""[plan 1103] 第一貢獻同步版（A2）逐折對照表：同步 vs StyleDDG（對外比法）、同步 vs 非同步（同為 BN 平均B）。

口徑（research/1103_contrib1_sync_version_3fold_plan.md §12.3）：
  StyleDDG   ＝ energy 讀出、BN 原樣（對外靶；0909_zperp_oscr_4fold.csv 的 p2p_np9／final）
  非同步 A0  ＝ 面讀出 zperp、評估端 BN 平均B（0909 CSV 的 p2p_np9_avgbn／final；OSA 用 results/osa/<折>.npz 的 ours/avg）
  同步 A2    ＝ 面讀出 zperp、**單次合併**：osdg_eval 的 epoch_200＋--avg_bn 列（GPU 上與訓練收尾合併逐位相同，1103 驗）；
               OSA 用 results/osa/sync_A2/<折>.npz 的 ours/raw（final 已在訓練收尾合併過；ours/avg 是雙重合併，只差捨入）
每折先用同一支程式重算登記值當錨點（§A 現行靶表、0905 §4.4／§4.7），對不上就標 ❌。
判讀級別沿用 0918 §7（事前寫死）：泛化 |Δ|≤1pp 無法分辨／≤3 灰帶／>3 真差異候選（需四折同號）；AUROC 單位 0.01。
用法：./venv_matcha/bin/python scripts/posthoc/contrib1_sync_fold_table.py [--folds sketch art_painting ...]
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.chdir(ROOT)
_argv, sys.argv = sys.argv, [sys.argv[0]]
from osa_4fold_summary import per_node  # noqa: E402
sys.argv = _argv

PACS = ["art_painting", "cartoon", "photo", "sketch"]
REG_CSV = "research/V1_baseline/0909_zperp_oscr_4fold.csv"
SUF = "_osdg_excl_person_seed2026_topo1234"
NAME = {"B": lambda f: f"v1_stage2_leave_{f}_nodiff{SUF}",
        "A0": lambda f: f"v1_stage2_leave_{f}_p1a_async_const_tau1e-5_style_nodiff_aggbn{SUF}" + ("_fix" if f == "cartoon" else ""),
        "A2": lambda f: f"v1_stage2_leave_{f}_p1a_nodiff_aggbn{SUF}"}
# 登記值（§A 現行靶表；0905 §4.4 面 AUROC；0905 §4.7／第 1195 行 OSA）
ANCHOR = {"B": {"gen": [76.70, 71.53, 85.96, 70.74], "auroc": [0.7378, 0.8016, 0.8311, 0.8640],
                "oscr": [0.6314, 0.6294, 0.7775, 0.6441], "osa": [53.54, 56.74, 73.99, 54.75]},
          "A0": {"gen": [84.47, 79.21, 90.65, 72.38], "auroc": [0.8055, 0.8281, 0.9094, 0.7966],
                 "osa": [67.53, 67.20, 80.49, 57.66]}}
PI = 0.2


def osa(r):
    return ((1 - PI) * r["a_id"] + PI * r["r_ood"]) * 100


def rows9(path, run, topo, ckpt, sf):
    rs = [r for r in csv.DictReader(open(path)) if r["run"] == run and r["topo"] == topo and r["ckpt"] == ckpt
          and r["score_fn"] == sf]
    return rs if len(rs) == 9 else None


def m(rs, k):
    return sum(float(r[k]) for r in rs) / len(rs)


def level(d, unit):
    a = abs(d) / unit
    return "無法分辨" if a <= 1 else ("灰帶" if a <= 3 else "真差異候選")


def fold_numbers(f, osdg_new, raw_new, npz_dir):
    i = PACS.index(f)
    out, notes = {}, []
    b = rows9(REG_CSV, NAME["B"](f), "p2p_np9", "final", "energy")
    zb = np.load(f"results/osa/{f}.npz", allow_pickle=True)
    out["B"] = dict(gen=m(b, "closed_acc") * 100, auroc=m(b, "det_auroc"), oscr=m(b, "oscr"),
                    osa=osa(per_node(zb, "baseline", "raw", "energy")))
    a0z = rows9(REG_CSV, NAME["A0"](f), "p2p_np9_avgbn", "final", "zperp")
    out["A0"] = dict(gen=m(a0z, "closed_acc") * 100, auroc=m(a0z, "det_auroc"), oscr=m(a0z, "oscr"),
                     osa=osa(per_node(zb, "ours", "avg", "zperp")))
    for arm in ("B", "A0"):
        for k, vals in ANCHOR[arm].items():
            ok = abs(out[arm][k] - vals[i]) < (0.006 if k in ("gen", "osa") else 0.00006)
            if not ok:
                notes.append(f"❌ 錨點 {arm}.{k} 重算 {out[arm][k]:.4f} ≠ 登記 {vals[i]}")
    a2 = rows9(osdg_new, NAME["A2"](f), "p2p_np9_avgbn", "epoch_200", "zperp")
    npz = os.path.join(npz_dir, f"{f}.npz")
    if a2 is None or not os.path.exists(npz):
        return out, notes + ["（同步 A2 尚無評估結果）"], None
    z2 = np.load(npz, allow_pickle=True)
    bb = [k for k in zb.files if k.startswith("baseline")]
    same = sum(k in z2.files and np.array_equal(z2[k], zb[k]) for k in bb)
    if same != len(bb):
        notes.append(f"❌ 同步 npz 的 StyleDDG 半邊只有 {same}/{len(bb)} 欄與登記相同")
    out["A2"] = dict(gen=m(a2, "closed_acc") * 100, auroc=m(a2, "det_auroc"), oscr=m(a2, "oscr"),
                     osa=osa(per_node(z2, "ours", "raw", "zperp")))
    ext = {}
    for arm, path, run, topo, ck in [("A0", REG_CSV, NAME["A0"](f), "p2p_np9_avgbn", "final"),
                                     ("A2", osdg_new, NAME["A2"](f), "p2p_np9_avgbn", "epoch_200")]:
        p = rows9(path, run, topo, ck, "proto_angle")
        e = rows9(path, run, topo, ck, "energy")
        ext[arm] = dict(point=m(p, "det_auroc") if p else None, energy=m(e, "det_auroc") if e else None)
    raw = [r for r in csv.DictReader(open(raw_new)) if r["run"] == NAME["A2"](f) and r["score_fn"] == "zperp"]
    acc = [float(r["closed_acc"]) * 100 for r in sorted(raw, key=lambda r: r["node"])]
    r5 = dict(nodes=acc, mean=sum(acc) / len(acc), rng=max(acc) - min(acc)) if len(acc) == 9 else None
    return out, notes, dict(ext=ext, r5=r5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=PACS)
    ap.add_argument("--osdg_csv", default="results/topology_axis/1103_sync_A2_osdg_avgbnB.csv")
    ap.add_argument("--raw_csv", default="results/topology_axis/1103_sync_A2_osdg_rawbn_epoch200.csv")
    ap.add_argument("--npz_dir", default="results/osa/sync_A2")
    a = ap.parse_args()
    K = [("gen", "泛化", 1.0, "{:.2f}", "{:+.2f}"), ("auroc", "部署AUROC", 0.01, "{:.4f}", "{:+.4f}"),
         ("oscr", "OSCR", 0.01, "{:.4f}", "{:+.4f}"), ("osa", "OSA20", 1.0, "{:.2f}", "{:+.2f}")]
    LAB = {"B": "StyleDDG＋energy／原樣", "A0": "我方非同步／平均B", "A2": "我方同步／單次合併"}
    done = {}
    for f in a.folds:
        out, notes, extra = fold_numbers(f, a.osdg_csv, a.raw_csv, a.npz_dir)
        print(f"\n===== {f}（ep200、9 節點平均；我方讀出＝面 zperp）=====")
        print(f"{'':24s}" + "".join(f"{lab:>12s}" for _, lab, *_ in K))
        for arm in ("B", "A0", "A2"):
            if arm in out:
                print(f"{LAB[arm]:24s}" + "".join(f"{fmt.format(out[arm][k]):>12s}" for k, _, _, fmt, _ in K))
        if "A2" in out:
            for lab, x, y in [("同步 − StyleDDG", "A2", "B"), ("非同步 − StyleDDG", "A0", "B"), ("同步 − 非同步", "A2", "A0")]:
                print(f"{lab:24s}" + "".join(f"{dfmt.format(out[x][k] - out[y][k]):>12s}" for k, _, _, _, dfmt in K)
                      + ("" if lab != "同步 − 非同步" else
                         f"   ← 級別：泛化 {level(out[x]['gen'] - out[y]['gen'], 1.0)}、AUROC {level(out[x]['auroc'] - out[y]['auroc'], 0.01)}"))
            e = extra["ext"]
            print(f"  參考：點讀出 AUROC 非同步 {e['A0']['point']:.4f}／同步 {e['A2']['point']:.4f}；"
                  f"我方模型 energy AUROC 非同步 {e['A0']['energy']:.4f}／同步 {e['A2']['energy']:.4f}")
            if extra["r5"]:
                r = extra["r5"]
                print(f"  R5 融合前（epoch_200、各節點自己的 BN）逐節點泛化 {[round(x, 2) for x in r['nodes']]}："
                      f"平均 {r['mean']:.2f}、極差 {r['rng']:.2f}pp → 融合後 {out['A2']['gen']:.2f}（{out['A2']['gen'] - r['mean']:+.2f}pp）")
            done[f] = out
        print("  錨點：" + ("全部重現登記值 ✅" if not [n for n in notes if n.startswith("❌")] else ""))
        for n in notes:
            print("  " + n)
    if len(done) == 4:
        print("\n===== 四折平均與勝場（同步 vs StyleDDG；非同步 vs StyleDDG）=====")
        for k, lab, unit, fmt, dfmt in K:
            d2 = [done[f]["A2"][k] - done[f]["B"][k] for f in PACS]
            d0 = [done[f]["A0"][k] - done[f]["B"][k] for f in PACS]
            dd = [done[f]["A2"][k] - done[f]["A0"][k] for f in PACS]
            print(f"{lab:8s} 同步−StyleDDG 平均 {dfmt.format(np.mean(d2))}、勝 {sum(x > 0 for x in d2)}:4｜"
                  f"非同步−StyleDDG 平均 {dfmt.format(np.mean(d0))}、勝 {sum(x > 0 for x in d0)}:4｜"
                  f"同步−非同步 平均 {dfmt.format(np.mean(dd))}（{level(np.mean(dd), unit)}）、同號 "
                  f"{max(sum(x > 0 for x in dd), sum(x < 0 for x in dd))}/4")


if __name__ == "__main__":
    main()
