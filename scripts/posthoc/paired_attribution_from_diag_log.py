#!/usr/bin/env python3
"""從三臂 diag 批次 log 產生配對分帳表（判準見 0906_art_two_arms_preregistration.md）。

用法: python3 paired_attribution_from_diag_log.py <log> [--arms lam0 conly full]

⚠️ 所有敘述性數字一律從變數帶出、不手打（2026-09-06 兩次「在能算的地方用了猜的」的對策）。
⚠️ r 的主定義＝逐節點 d'128/d'512 → 逐畫風平均（恆等式只在逐節點層級成立）。
⚠️ 有效 n = 3 畫風、不是 9 節點（同畫風節點特徵餘弦 >0.9998）。
"""
import re, sys, argparse, statistics as st

NODE = re.compile(
    r"node_(\d) \((\w+)\s*\) 512:d'=([\d.]+) AUROC=([\d.]+).*?"
    r"128:d'=([\d.]+) AUROC=([\d.]+).*?energy:d'=([\d.]+) AUROC=([\d.]+)")

def parse(path):
    txt = open(path, encoding="utf-8").read()
    parts = re.split(r"^##########\s+(\S+)\s+ckpt=(\S+)", txt, flags=re.M)
    out = {}
    for i in range(1, len(parts), 3):
        tag, ckpt, body = parts[i], parts[i+1], parts[i+2]
        rows = [dict(node=int(m[0]), style=m[1], d512=float(m[2]), au512=float(m[3]),
                     d128=float(m[4]), au128=float(m[5]), dE=float(m[6]), auE=float(m[7]))
                for m in NODE.findall(body)]
        if rows:
            out[tag] = dict(ckpt=ckpt, rows=rows)
    return out

def style_means(v):
    """9 節點 → 3 畫風平均（假設 3 節點/畫風、依 node 順序分組）。"""
    return [st.mean(v[0:3]), st.mean(v[3:6]), st.mean(v[6:9])]

def paired(a, b, key):
    """b − a 的配對差，回傳 (逐節點, 逐畫風, mean, SE, t, 同號數)。"""
    v = [b[j][key] - a[j][key] for j in range(len(a))]
    sm = style_means(v); m = st.mean(sm)
    se = st.stdev(sm) / len(sm) ** 0.5
    same = max(sum(1 for x in v if x > 0), sum(1 for x in v if x < 0))
    return v, sm, m, se, (m / se if se else float("nan")), same

def r_per_node(rows):
    return [x["d128"] / x["d512"] for x in rows]

def report(D, arms):
    lam0, conly, full = (D[a]["rows"] for a in arms)
    print(f"# 配對分帳（{arms[0]} → {arms[1]} → {arms[2]}）\n")
    for a in arms:
        print(f"  {a:6s} ckpt={D[a]['ckpt']}")
    print()

    print("## 各臂 node-mean（原始值）")
    hdr = f"{'arm':7s} {'512':>8s} {'128':>8s} {'energy':>8s} {'d512':>7s} {'d128':>7s} {'r=d128/d512':>12s}"
    print(hdr)
    for a in arms:
        R = D[a]["rows"]
        r = st.mean(style_means(r_per_node(R)))
        print(f"{a:7s} {st.mean([x['au512'] for x in R]):8.4f} "
              f"{st.mean([x['au128'] for x in R]):8.4f} {st.mean([x['auE'] for x in R]):8.4f} "
              f"{st.mean([x['d512'] for x in R]):7.3f} {st.mean([x['d128'] for x in R]):7.3f} {r:12.4f}")
    print()

    for scale, key in [("部署 AUROC（主判準）", "au128"), ("energy（投影層以外）", "auE"),
                       ("512（投影層以外）", "au512")]:
        print(f"## {scale}")
        for nm, (x, y) in [("comp", (lam0, conly)), ("disp", (conly, full)), ("總", (lam0, full))]:
            v, sm, m, se, t, same = paired(x, y, key)
            print(f"  {nm:4s} 逐節點 {' '.join(f'{q:+.4f}' for q in v)}")
            print(f"       逐畫風 {' '.join(f'{q:+.4f}' for q in sm)} | mean {m:+.4f} "
                  f"SE {se:.4f} 3SE {3*se:.4f} t={t:6.1f} 同號 {same}/9")
        print()

    print("## r 尺度（交叉檢查；主定義＝逐節點 d128/d512 → 逐畫風平均）")
    RR = {a: r_per_node(D[a]["rows"]) for a in arms}
    for nm, (x, y) in [("comp", (arms[0], arms[1])), ("disp", (arms[1], arms[2])), ("總", (arms[0], arms[2]))]:
        v = [RR[y][j] - RR[x][j] for j in range(9)]
        sm = style_means(v); m = st.mean(sm); se = st.stdev(sm) / 3 ** 0.5
        same = max(sum(1 for q in v if q > 0), sum(1 for q in v if q < 0))
        print(f"  {nm:4s} 逐畫風 {' '.join(f'{q:+.4f}' for q in sm)} | mean {m:+.4f} "
              f"SE {se:.4f} 3SE {3*se:.4f} t={m/se:6.1f} 同號 {same}/9")
    print()

    # ── 判準裁決（A1/A2/A3；門檻見登記表 §4／§4.1，全部事前寫死）
    _, _, m, se, _, same = paired(conly, full, "au128")
    lo, hi, cart = 0.0282, 0.0564, 0.0318
    print("## 判準裁決（AUROC 尺度、disp 主軸）")
    print(f"  disp 貢獻 = {m:+.4f}   3SE = {3*se:.4f}   逐節點同號 {same}/9")
    if m > 3 * se:
        print("  ⇒ 【A1】disp 在 art 有幫助 ⇒「保住好處」是真命題 ⇒ 條件化，不可直接拿掉")
    elif m < -3 * se:
        print("  ⇒ 【A3】disp 在 art 也有害 ⇒ 拿掉幾乎無代價")
    else:
        print(f"  ⇒ 【A2】本輪檢力下無法與零區分（不是「沒作用」）")
        print(f"     可能被漏掉的效應上限 ±{3*se:.4f}；cartoon 已知正效應 +{cart:.4f}")
        x = abs(m)
        if x < lo:
            print(f"     點估計 {x:.4f} < {lo} ⇒ 即使 photo 與 art 同量級，拿掉仍淨賺 ⇒ 可寫「代價小」")
        elif x <= hi:
            print(f"     點估計 {x:.4f} ∈ [{lo}, {hi}] ⇒ 符號取決於未測的 photo ⇒ 🚨 不下修法結論")
        else:
            print(f"     點估計 {x:.4f} > {hi} ⇒ 四折平均必定淨賠 ⇒ 走 A1 的行動")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log"); p.add_argument("--arms", nargs=3, default=["lam0", "conly", "full"])
    a = p.parse_args()
    D = parse(a.log)
    miss = [x for x in a.arms if x not in D]
    if miss:
        sys.exit(f"log 缺這些臂: {miss}（有的是 {list(D)}）")
    report(D, a.arms)
