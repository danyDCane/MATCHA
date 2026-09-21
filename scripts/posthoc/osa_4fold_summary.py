"""四 fold 開集準確率 OSA 彙總——「逐張去向」表 + 三種口徑的 Δ。

輸入：scripts/open_set_accuracy.py --stage infer 產出的 <fold>.npz（四個）。
指標血緣：OSA ＝ 1 −〔SCOD(ICLR'24) 聯合風險，改用 AUGRC(NeurIPS'24) 的 generalized 分母〕。
⚠️ 舊稱「wild accuracy」已停用（與文獻中訓練用的 wild data 語義衝突）。
呈現：假想 1000 張測試流（π 由 --pi 指定，預設 20% ⇒ 已知 800、未知 200），
      把每一張圖的下場逐格攤開——這是最不需要解釋就看得懂的形式（dany 2026-09-04）。
      張數是期望值（比例 × 1000），不是重抽樣 ⇒ 有小數、四捨五入後一列可能差 1。

逐格信心度（2026-09-17 併入原 0905 §5.3）：每格附「六類頭最大類別機率」的平均。
⚠️ 口徑＝**每張圖算一次**：9 節點的判決倒進同一池再平均（`cell_pool`），**不是**各節點先平均再 9 節點等權。
   後者會讓「該格只有 10 張」的節點與「該格有 82 張」的節點同權 ⇒ 漏放 person 的信心度被高估
   （sketch 我方：節點等權 0.824 vs 每張一次 0.770），且某節點該格為 0 張時無定義。
   四 fold 平均列：張數＝四 fold 每千張張數等權平均；信心度＝以該格四 fold 張數加權（＝四個 1000 張倒進同一池）。

⚠️ 三種口徑必須分開報（同口徑才是可辯護的比較）：
   ① 都不做部署前 BN 匯聚（原樣 vs 原樣）      ← 我方主張「不需後處理」的證據
   ② 都做部署前 BN 匯聚（平均B vs 平均B）      ← 最嚴格的同口徑對照
   ③ 我方平均B vs baseline 原樣                ← 不對等，僅供對照，引用必須標明
"""
import os
import sys
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

PACS = ["art_painting", "cartoon", "photo", "sketch"]
SHORT = {"art_painting": "art", "cartoon": "cartoon", "photo": "photo", "sketch": "sketch"}
N_NODES = 9
# 0903 §3（baseline closed_acc、BN 原樣、ep200）——不經 BN 平均，未受 osdg_eval BN bug 影響 ⇒ 硬錨
ANCHOR_BASE_ACC = {"cartoon": 71.53, "art_painting": 76.70, "photo": 85.96, "sketch": 70.74}


def per_node(z, tag, bn, readout, q=0.95):
    from sklearn.metrics import roc_auc_score
    out = []
    for i in range(N_NODES):
        p = f"{tag}__{bn}__{i}"
        if f"{p}__src_{readout}" not in z:
            return None
        s_src, s_k, s_u = z[f"{p}__src_{readout}"], z[f"{p}__tgt_known_{readout}"], z[f"{p}__tgt_unk_{readout}"]
        corr = z[f"{p}__tgt_known_correct"] > 0
        tau = float(np.quantile(s_src, q))
        acc_k = s_k <= tau
        out.append(dict(a_id=float((acc_k & corr).mean()), r_ood=float((s_u > tau).mean()),
                        A=float(corr.mean()), fpr=float((~acc_k).mean()),
                        auroc=float(roc_auc_score(np.r_[np.zeros(len(s_k)), np.ones(len(s_u))],
                                                  np.r_[s_k, s_u])),
                        n_known=len(s_k), n_unk=len(s_u)))
    return {k: float(np.mean([r[k] for r in out])) for k in
            ["a_id", "r_ood", "A", "fpr", "auroc"]} | {"n_known": out[0]["n_known"], "n_unk": out[0]["n_unk"]}


CELLS = ["rej_ok", "rej_wrong", "pass_wrong", "pass_ok", "unk_rej", "unk_pass"]
CELL_LAB = {"rej_ok": "誤拒·原本分對", "rej_wrong": "誤拒·原本分錯", "pass_wrong": "放行分錯",
            "pass_ok": "放行分對✓", "unk_rej": "未知:拒絕✓", "unk_pass": "未知:放行"}


def cell_pool(z, tag, bn, readout, q=0.95):
    """一個 fold × 一個臂：9 節點的判決倒進同一池（每張圖算一次）。

    readout=None ⇒ 無檢測（全部放行）。
    回傳 {cell: (比例, 平均信心度)}；比例的分母＝全部已知（rej_*/pass_*）或全部未知（unk_*）。
    ⚠️ 各節點的目標域測試集相同 ⇒ 比例與「逐節點比例再平均」（per_node）逐位相同，只有信心度的平均方式不同。
    存檔的 msp 已定向為「高＝像 OOD」（open_set_accuracy.py 存的是 −max softmax）⇒ 這裡翻號回信心度。
    """
    n = dict.fromkeys(CELLS, 0)
    s = dict.fromkeys(CELLS, 0.0)
    nk = nu = 0
    for i in range(N_NODES):
        p = f"{tag}__{bn}__{i}"
        pk = -z[f"{p}__tgt_known_msp"].astype(np.float64)
        pu = -z[f"{p}__tgt_unk_msp"].astype(np.float64)
        corr = z[f"{p}__tgt_known_correct"] > 0
        if readout is None:
            rk, ru = np.zeros(len(pk), bool), np.zeros(len(pu), bool)
        else:
            tau = float(np.quantile(z[f"{p}__src_{readout}"], q))
            rk, ru = z[f"{p}__tgt_known_{readout}"] > tau, z[f"{p}__tgt_unk_{readout}"] > tau
        for c, (v, m) in {"rej_ok": (pk, rk & corr), "rej_wrong": (pk, rk & ~corr),
                          "pass_wrong": (pk, ~rk & ~corr), "pass_ok": (pk, ~rk & corr),
                          "unk_rej": (pu, ru), "unk_pass": (pu, ~ru)}.items():
            n[c] += int(m.sum())
            s[c] += float(v[m].sum())
        nk += len(pk)
        nu += len(pu)
    return {c: (n[c] / (nu if c.startswith("unk") else nk), s[c] / n[c] if n[c] else np.nan) for c in CELLS}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default="results/osa")
    ap.add_argument("--pi", type=float, default=0.2)
    ap.add_argument("--out_csv", default="results/osa/summary_4fold.csv")
    ap.add_argument("--md_out", default="results/osa/summary_4fold_cells.md",
                    help="逐張去向表（含逐格信心度）的 markdown，直接貼進 0905 §3")
    a = ap.parse_args()
    PI, NK, NU = a.pi, int(1000 * (1 - a.pi)), int(1000 * a.pi)

    COMBOS = [("nodet", "baseline", "raw", ""), ("nodet", "baseline", "avg", ""),
              ("nodet", "ours", "raw", ""), ("nodet", "ours", "avg", ""),
              ("det", "baseline", "raw", "energy"), ("det", "baseline", "avg", "energy"),
              ("det", "ours", "raw", "proto"), ("det", "ours", "avg", "proto"),
              ("det", "ours", "avg", "zperp"),
              ("det", "ours", "raw", "energy"), ("det", "ours", "avg", "energy")]
    LAB = {("nodet", "baseline", "raw", ""): "StyleDDG 無檢測/原樣",
           ("nodet", "baseline", "avg", ""): "StyleDDG 無檢測/平均B",
           ("nodet", "ours", "raw", ""): "我方 無檢測/原樣",
           ("nodet", "ours", "avg", ""): "我方 無檢測/平均B",
           ("det", "baseline", "raw", "energy"): "StyleDDG+energy/原樣",
           ("det", "baseline", "avg", "energy"): "StyleDDG+energy/平均B",
           ("det", "ours", "raw", "proto"): "我方+原型/原樣",
           ("det", "ours", "avg", "proto"): "我方+原型/平均B",
           ("det", "ours", "avg", "zperp"): "我方+‖z⊥‖面/平均B",
           ("det", "ours", "raw", "energy"): "我方+energy/原樣",
           ("det", "ours", "avg", "energy"): "我方+energy/平均B"}

    D, folds = {}, []
    for f in PACS:
        p = os.path.join(a.npz_dir, f"{f}.npz")
        if not os.path.exists(p):
            print(f"[skip] 缺 {p}")
            continue
        z = np.load(p, allow_pickle=True)
        folds.append(f)
        for c in COMBOS:
            r = per_node(z, c[1], c[2], c[3] or "energy")
            if r is None:
                continue
            if c[0] == "nodet":                     # 無檢測：全部放行，未知全錯
                r = dict(r, a_id=r["A"], r_ood=0.0, fpr=0.0)
            r["cells"] = cell_pool(z, c[1], c[2], c[3] if c[0] == "det" else None)
            # 自檢：cell_pool 的比例必須與 per_node 逐位相同（同一份判決、只是彙總路徑不同）
            assert abs(r["cells"]["pass_ok"][0] - r["a_id"]) < 1e-9 and abs(r["cells"]["unk_rej"][0] - r["r_ood"]) < 1e-9
            D[(f,) + c] = r

    def cells(r):
        """{cell: (每千張張數, 信心度)}"""
        return {c: (fr * (NU if c.startswith("unk") else NK), cf) for c, (fr, cf) in r["cells"].items()}

    def avg_cells(rs):
        """四 fold 平均：張數等權平均；信心度以該格各 fold 張數加權。"""
        out = {}
        for c in CELLS:
            ns = np.array([cells(r)[c][0] for r in rs])
            cs = np.array([cells(r)[c][1] for r in rs])
            ok = ns > 0
            out[c] = (float(ns.mean()), float((ns[ok] * cs[ok]).sum() / ns[ok].sum()) if ok.any() else np.nan)
        return out

    def md_row(lab, cl):
        fmt = lambda n, cf: f"{n:.0f}" if round(n) == 0 else f"{n:.0f} ({cf:.3f})"
        tot = cl["pass_ok"][0] + cl["unk_rej"][0]
        return f"| {lab} | " + " | ".join(fmt(*cl[c]) for c in CELLS) + f" | {tot:.0f} | {tot / 10:.1f}% |"

    MD_HDR = ("| 方法 | " + " | ".join(CELL_LAB[c] for c in CELLS) + " | 總對 | acc |\n|---|"
              + "---:|" * (len(CELLS) + 2))
    md = []

    W = 108
    print("=" * W)
    print(f"§0 自檢：baseline closed_acc（BN 原樣）對 0903 §3 錨點")
    print("=" * W)
    for f in folds:
        r = D.get((f, "nodet", "baseline", "raw", ""))
        if r:
            v, an = r["A"] * 100, ANCHOR_BASE_ACC[f]
            flag = "✅" if abs(v - an) < 0.1 else "🚨"
            print(f"  {SHORT[f]:<9}{v:>8.2f}%   錨 {an:.2f}%   差 {v-an:+.2f}pp  {flag}"
                  f"   (已知 {r['n_known']} / person {r['n_unk']})")

    print("\n" + "=" * W)
    print(f"§1 逐張去向（每 fold 假想 1000 張，π={PI:.0%} ⇒ 已知 {NK}、未知 {NU}）")
    print("=" * W)
    hdr = (f"  {'方法':<24}{'誤拒·分對':>16}{'誤拒·分錯':>16}{'放行分錯':>16}{'放行分對✓':>16}"
           f"{'未知:拒絕✓':>16}{'未知:放行':>16}{'總對':>8}{'acc':>9}")
    KEY = [("nodet", "baseline", "raw", ""),
           ("det", "baseline", "raw", "energy"), ("det", "baseline", "avg", "energy"),
           ("nodet", "ours", "raw", ""), ("nodet", "ours", "avg", ""),
           ("det", "ours", "raw", "proto"), ("det", "ours", "avg", "proto"),
           ("det", "ours", "avg", "zperp"),
           ("det", "ours", "raw", "energy"), ("det", "ours", "avg", "energy")]
    pr = lambda lab, cl: print(f"  {lab:<24}" + "".join(
        f"{cl[c][0]:>7.0f}" + (f" ({cl[c][1]:.3f})" if round(cl[c][0]) else " " * 8) for c in CELLS)
        + f"{cl['pass_ok'][0] + cl['unk_rej'][0]:>8.0f}{(cl['pass_ok'][0] + cl['unk_rej'][0]) / 10:>8.1f}%")
    rows_cells = []
    for f in folds:
        r0 = D[(f, "nodet", "baseline", "raw", "")]
        print(f"\n  ── leave-out = {SHORT[f]}"
              f"（實際測試集：已知 6 類 {r0['n_known']} 張、person {r0['n_unk']} 張，"
              f"實際未知比例 {r0['n_unk']/(r0['n_known']+r0['n_unk']):.1%}）──")
        print(hdr)
        md.append(f"\n### leave-out = {f}（已知 {r0['n_known']}／person {r0['n_unk']}，"
                  f"實際未知比例 {r0['n_unk']/(r0['n_known']+r0['n_unk']):.1%}）\n\n{MD_HDR}")
        for c in KEY:
            r = D.get((f,) + c)
            if not r:
                continue
            if c == ("nodet", "ours", "raw", ""):
                print(f"  {'-'*100}")
            cl = cells(r)
            pr(LAB[c], cl)
            md.append(md_row(LAB[c], cl))
            tot = cl["pass_ok"][0] + cl["unk_rej"][0]
            rows_cells.append(dict(fold=f, method=LAB[c], n_known=r["n_known"], n_unk=r["n_unk"],
                                   known_rejected=round(cl["rej_ok"][0] + cl["rej_wrong"][0]),
                                   known_rejected_correct=round(cl["rej_ok"][0]),
                                   known_rejected_wrong=round(cl["rej_wrong"][0]),
                                   known_pass_wrong=round(cl["pass_wrong"][0]),
                                   known_pass_correct=round(cl["pass_ok"][0]), unk_rejected=round(cl["unk_rej"][0]),
                                   unk_passed=round(cl["unk_pass"][0]), total_correct=round(tot),
                                   osa=round(tot / 10, 2), auroc=round(r["auroc"], 4),
                                   **{f"conf_{k}": (round(cl[k][1], 4) if round(cl[k][0]) else "") for k in CELLS}))

    print("\n" + "=" * W)
    print(f"§2 ★ 四 fold 平均的逐張去向（π={PI:.0%}）")
    print("=" * W)
    print(hdr)
    md.append(f"\n### 四 fold 平均\n\n{MD_HDR}")
    for c in KEY:
        rs = [D[(f,) + c] for f in folds if (f,) + c in D]
        if len(rs) != len(folds):
            continue
        cl = avg_cells(rs)
        pr(LAB[c], cl)
        md.append(md_row(LAB[c], cl))

    print("\n" + "=" * W)
    print(f"§3 ★★ 我方 vs StyleDDG：OSA 的 Δ（π={PI:.0%}，逐 fold ＋ 平均）")
    print("=" * W)
    wa = lambda r: ((1 - PI) * r["a_id"] + PI * r["r_ood"]) * 100
    CMP = [("① 同口徑・都不做 BN 匯聚", ("det", "ours", "raw", "proto"), ("det", "baseline", "raw", "energy")),
           ("② 同口徑・都做 BN 匯聚", ("det", "ours", "avg", "proto"), ("det", "baseline", "avg", "energy")),
           ("③ ⚠️不對等・我方平均B vs baseline 原樣", ("det", "ours", "avg", "proto"), ("det", "baseline", "raw", "energy")),
           ("④ 參考・我方換 energy 讀出(同①口徑)", ("det", "ours", "raw", "energy"), ("det", "baseline", "raw", "energy"))]
    print(f"  {'口徑':<38}" + "".join(f"{SHORT[f]:>10}" for f in folds) + f"{'平均Δ':>10}{'勝場':>7}")
    rows_csv = []
    for name, ok, bk in CMP:
        ds = []
        for f in folds:
            ro, rb = D.get((f,) + ok), D.get((f,) + bk)
            ds.append(wa(ro) - wa(rb) if ro and rb else np.nan)
            rows_csv.append(dict(compare=name, fold=f, ours=round(wa(ro), 2) if ro else "",
                                 baseline=round(wa(rb), 2) if rb else "",
                                 delta=round(ds[-1], 2) if ro and rb else ""))
        w = sum(1 for d in ds if d > 0)
        print(f"  {name:<38}" + "".join(f"{d:>+10.2f}" for d in ds)
              + f"{np.nanmean(ds):>+10.2f}{w}/{len(ds):>6}")

    print("\n" + "=" * W)
    print(f"§4 逐 fold OSA 絕對值（π={PI:.0%}）")
    print("=" * W)
    print(f"  {'方法':<24}" + "".join(f"{SHORT[f]:>10}" for f in folds) + f"{'平均':>10}")
    for c in KEY:
        vs = [wa(D[(f,) + c]) for f in folds if (f,) + c in D]
        if len(vs) == len(folds):
            print(f"  {LAB[c]:<24}" + "".join(f"{v:>10.2f}" for v in vs) + f"{np.mean(vs):>10.2f}")

    print("\n" + "=" * W)
    print("§5 檢測品質（部署 AUROC，四 fold）——與 OSA 分開看")
    print("=" * W)
    print(f"  {'方法':<24}" + "".join(f"{SHORT[f]:>10}" for f in folds) + f"{'平均':>10}")
    for c in KEY[1:]:
        vs = [D[(f,) + c]["auroc"] for f in folds if (f,) + c in D]
        if len(vs) == len(folds):
            print(f"  {LAB[c]:<24}" + "".join(f"{v:>10.4f}" for v in vs) + f"{np.mean(vs):>10.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.md_out)) or ".", exist_ok=True)
    with open(a.md_out, "w") as fh:
        fh.write("\n".join(md).lstrip() + "\n")
    print(f"\n  逐張表 markdown → {a.md_out}")
    cells_csv = a.out_csv.replace(".csv", "_cells.csv")
    with open(cells_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_cells[0].keys()))
        w.writeheader()
        w.writerows(rows_cells)
    print(f"\n  逐張表 → {cells_csv}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)) or ".", exist_ok=True)
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\n  → {a.out_csv}")


if __name__ == "__main__":
    main()
