"""
V1 驗證薄收集器：把 in-training 診斷（D1–D4 CSV）與最終 OOD（test.sh 輸出）彙成一張結論表。

只用標準庫（csv），不引入 pandas，避免額外依賴。

用法：
    python -m dood.collect_results \
        --diag_dir exp_result_xxx/agg_diag \
        --results_csv results/domain_ood_scores_eps_mse.csv \
        --stage stage1_graphid-1_leave_art \
        --out research/0522_V1_stage1_summary.md

results_csv 可省略（訓練尚未跑 test 時只彙整診斷）。
"""

import os
import csv
import argparse
from collections import defaultdict


def _read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def summarize_d1_d4(rows):
    """D1/D4 隨輪變化的摘要：最終值、最大值、首尾趨勢、NaN 計數。"""
    if not rows:
        return "（無 D1/D4 資料）\n"
    def col(name):
        return [_f(r.get(name)) for r in rows]
    d1g, d1i, d1e = col("d1_global"), col("d1_intra"), col("d1_inter")
    d4m = col("d4_dL_mean")
    n_nan = sum(int(_f(r.get("has_nan"), 0)) for r in rows)
    first, last = rows[0], rows[-1]
    out = []
    out.append(f"- 通訊輪數：{len(rows)}（round {first.get('round')} → {last.get('round')}，epoch {first.get('epoch')} → {last.get('epoch')}）")
    out.append(f"- **D1 共識發散度** global：末值={_f(last.get('d1_global')):.3e}，全程最大={max(d1g):.3e}，全程最小={min(d1g):.3e}")
    out.append(f"  - intra-domain 末值={_f(last.get('d1_intra')):.3e}；inter-domain 末值={_f(last.get('d1_inter')):.3e}")
    out.append(f"- **D4 ΔL（聚合前後）** 末值={_f(last.get('d4_dL_mean')):.3e}；全程 |max|={max((abs(v) for v in d4m if v == v), default=float('nan')):.3e}")
    out.append(f"  - ΔL>0 的輪數佔比：{sum(1 for v in d4m if v == v and v > 0)}/{sum(1 for v in d4m if v == v)}")
    out.append(f"- NaN/Inf 標記輪數：{n_nan}")
    return "\n".join(out) + "\n"


def summarize_d2(rows):
    """D2：取最後一個 epoch 的 per-domain ID NLL。"""
    if not rows:
        return "（無 D2 資料）\n"
    last_epoch = max(int(_f(r.get("epoch"), 0)) for r in rows)
    dom_rows = [r for r in rows if r.get("scope") == "domain" and int(_f(r.get("epoch"), -1)) == last_epoch]
    out = [f"- 最後 epoch={last_epoch} 的 per-domain ID NLL（DSM loss）："]
    for r in sorted(dom_rows, key=lambda x: x.get("domain", "")):
        out.append(f"  - {r.get('domain')}: {_f(r.get('nll')):.4e}" + ("  ⚠NaN" if int(_f(r.get('has_nan'), 0)) else ""))
    return "\n".join(out) + "\n"


def summarize_d3(rows):
    """D3：取最後 epoch 的矩陣，算 intra-domain block 與 inter-domain block 平均。"""
    if not rows:
        return "（無 D3 資料）\n"
    last_epoch = max(int(_f(r.get("epoch"), 0)) for r in rows)
    mat = [r for r in rows if int(_f(r.get("epoch"), -1)) == last_epoch]
    intra, inter = [], []
    for r in mat:
        s = _f(r.get("score"))
        if s != s:
            continue
        (intra if r.get("i_domain") == r.get("j_domain") else inter).append(s)
    out = [f"- 最後 epoch={last_epoch} 跨節點異質性矩陣（{len(mat)} 格）："]
    if intra:
        out.append(f"  - intra-domain block 平均={sum(intra)/len(intra):.4e}（同 domain，應較低）")
    if inter:
        out.append(f"  - inter-domain block 平均={sum(inter)/len(inter):.4e}（跨 domain）")
    if intra and inter:
        ratio = (sum(inter)/len(inter)) / (sum(intra)/len(intra) + 1e-12)
        out.append(f"  - inter/intra 比值={ratio:.2f}（>1 表示跨 domain 模型較不相容）")
    return "\n".join(out) + "\n"


def summarize_ood(rows):
    """test.sh 輸出：列出含 AUROC/FPR 的列。"""
    if not rows:
        return "（無 test.sh OOD 結果；訓練後再跑 test.sh）\n"
    out = ["| train_domain | test_domain | label | mean_score | auroc | fpr95 |", "|---|---|---|---|---|---|"]
    for r in rows:
        auroc = r.get("auroc", "") or ""
        fpr = r.get("fpr95", "") or ""
        out.append(f"| {r.get('train_domain','')} | {r.get('test_domain','')} | {r.get('label','')} | "
                   f"{_f(r.get('mean_score')):.4f} | {auroc} | {fpr} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag_dir", required=True, help="agg_diag 目錄（含 diag_D1_D4/D2/D3.csv）")
    ap.add_argument("--results_csv", default=None, help="test.sh 輸出的 domain_ood_scores_*.csv（可省略）")
    ap.add_argument("--stage", default="", help="階段標籤，寫進標題")
    ap.add_argument("--out", required=True, help="輸出 markdown 路徑")
    args = ap.parse_args()

    d14 = _read_csv(os.path.join(args.diag_dir, "diag_D1_D4.csv"))
    d2 = _read_csv(os.path.join(args.diag_dir, "diag_D2.csv"))
    d3 = _read_csv(os.path.join(args.diag_dir, "diag_D3.csv"))
    ood = _read_csv(args.results_csv)

    md = []
    md.append(f"# V1 驗證結論表 — {args.stage}\n")
    md.append("## D1 共識發散度 + D4 聚合前後 ΔL（in-training）\n")
    md.append(summarize_d1_d4(d14))
    md.append("\n## D2 聚合模型 ID NLL（in-training）\n")
    md.append(summarize_d2(d2))
    md.append("\n## D3 跨節點異質性矩陣（in-training）\n")
    md.append(summarize_d3(d3))
    md.append("\n## 最終 OOD（test.sh）\n")
    md.append(summarize_ood(ood))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(md))
    print(f"[collect_results] 已寫出 {args.out}")


if __name__ == "__main__":
    main()
