"""StyleDDG baseline vs 我們全套：4-fold 對照彙總（closed_acc / spread / 逐源域 / energy OSCR）。

用法：python scripts/posthoc/summarize_baseline_vs_ourfull.py <csv_dir>
csv 由 scripts/osdg_eval.py 產生，命名慣例 baseline_<fold>_osdg.csv / ourfull_<fold>_osdg.csv。
逐源域分組依 PACS 字母序排除 leave_out 後，每域 3 個節點（util.py 的 virtual-node 指派）。
"""
import csv, os, sys, statistics

PACS = ["art_painting", "cartoon", "photo", "sketch"]
SHORT = {"art_painting": "art", "cartoon": "cartoon", "photo": "photo", "sketch": "sketch"}
KEY = {"cartoon": "cartoon", "art_painting": "art", "photo": "photo", "sketch": "sketch"}


def sources(leave_out):
    """節點 i 的來源域：3 個源域各佔 3 個節點，順序同 PACS 字母序。"""
    avail = [d for d in PACS if d != leave_out]
    return [avail[i // 3] for i in range(9)]


def load(path, leave_out):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    ca = {r["node"]: float(r["closed_acc"]) for r in rows}
    v = [ca[f"node_{i}"] for i in range(9)]
    src = sources(leave_out)
    grp = {s: statistics.mean([v[i] for i in range(9) if src[i] == s])
           for s in dict.fromkeys(src)}
    en = statistics.mean(float(r["oscr"]) for r in rows if r["score_fn"] == "energy")
    return dict(mean=statistics.mean(v), spread=max(v) - min(v), grp=grp, oscr=en)


def main(d):
    print(f"{'fold':13s}{'baseline':>10s}{'ourfull':>10s}{'Δ':>9s}"
          f"{'spread b→o':>16s}{'OSCR Δ':>10s}   逐源域 Δ")
    deltas = []
    for lo in PACS:
        k = KEY[lo]
        b = load(os.path.join(d, f"baseline_{k}_osdg.csv"), lo)
        o = load(os.path.join(d, f"ourfull_{k}_osdg.csv"), lo)
        if b is None:
            print(f"{lo:13s}  (baseline 未產出)"); continue
        if o is None:
            print(f"{lo:13s}{b['mean']*100:9.2f}%{'—':>10s}{'—':>9s}"
                  f"{b['spread']*100:12.1f}pp{'':>10s}   (全套未產出)"); continue
        deltas.append((o["mean"] - b["mean"]) * 100)
        gd = "  ".join(f"{SHORT[s]}{(o['grp'][s]-b['grp'][s])*100:+.1f}" for s in b["grp"])
        print(f"{lo:13s}{b['mean']*100:9.2f}%{o['mean']*100:9.2f}%"
              f"{(o['mean']-b['mean'])*100:+8.2f}pp"
              f"{b['spread']*100:8.1f}→{o['spread']*100:.1f}pp"
              f"{o['oscr']-b['oscr']:+10.4f}   {gd}")
    if deltas:
        print(f"\n  已完成 {len(deltas)} 個 fold 的 Δ 平均 = {statistics.mean(deltas):+.2f}pp"
              + (f"（全距 {max(deltas)-min(deltas):.2f}pp）" if len(deltas) > 1 else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
