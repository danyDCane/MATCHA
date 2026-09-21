"""開集準確率 OSA(π) —— 「OOD 檢測對泛化分類值多少」的部署側呈現。

🚨 **指標血緣（引用必標，勿宣稱自創）**：
    OSA(π) ＝ 1 − 〔SCOD 聯合風險（改用 generalized 分母）〕
  · **SCOD**（ICLR 2024，*Plugin estimators for selective classification with OOD detection*）
    定義聯合風險 `risk = (1−α)·RS + α·fpr`。本指標與之同構：π ↔ α、r_ood ＝ 1 − fpr。
  · **AUGRC / FD-Shifts**（NeurIPS 2024 Spotlight，*Overcoming Common Flaws in the Evaluation
    of Selective Classification Systems*）主張分母用「全體樣本」的 generalized risk
    `P(failure ∧ accepted)`、而非 SCOD 的 selective risk `P(failure | accepted)`。本指標的
    `a_id` 採 generalized 約定（分母＝全體已知類）。
  ⚠️ **不可再稱「wild accuracy」**：文獻中 wild data 指的是**訓練用的**無標籤混合資料
    （SCONE, ICML 2023 硬依賴外部 wild 資料），與本指標無關；我方為 outlier-free 設定
    （未知類別來自資料集內保留的 person），沿用該詞會招致「你們是否用了外部資料訓練」的誤解。

故事（dany 2026-09-04）：部署時測試流 = (1−π)·目標域已知類別 + π·未知類別(person)。
  - 沒有檢測器（SOTA StyleDDG）：每張 person 都必然分錯 ⇒ 準確率隨 π 線性下降。
    ⚠️ 這不是假設——0903 報告 §2 已實測：cartoon 的 acc.log ep200 node-mean 59.1724
       ÷ (1939/2344) = 71.5318 vs 直接算 71.5311（差 0.0007pp）⇒「person 100% 答錯」成立。
  - 有檢測器：超過門檻就拒絕，拒對了算一次正確決策。

指標：
    OSA(π) = (1−π)·a_id(τ) + π·r_ood(τ)
      a_id(τ)  = 目標域已知類別中「被放行 AND 分對」的比例   （被誤拒的已知類別＝損失，算錯）
      r_ood(τ) = 目標域 person 中「被拒絕」的比例
    無檢測基準：OSA(π) = (1−π)·A，A ＝ closed-set 6 類準確率

門檻 τ（可部署、不偷看目標域）：**該節點自己來源域測試集、只取已知類別**的分數 95% 分位。
  ⇒ 「我允許誤殺 5% 自己域的正常樣本」。每個節點各自算自己的 τ（部署上本來就是這樣）。

⚠️ π 是**比例加權的期望值**，不是實體重抽。person 張數固定（cartoon 405 張），
   實體湊 π 要子抽樣、會引入抽樣噪音；加權公式在數學上就是期望值，且 π 可連續掃。

⚠️ **固定門檻是刻意選擇，不是疏漏**：AUGRC 那篇批評固定閾值指標、主張聚合所有門檻。
   但部署時門檻必須事先定死，無法事後回頭挑。本指標要量的正是「事先定死一個合理門檻後
   實際會發生什麼」——而實測誤拒率從設計的 5% 變成 25–54%，這在門檻無關的指標上完全看不見。
   （門檻無關的 AUROC 與 OSCR 本專案同時照報，見 0905 報告 §4.4／§6.4。）

⚠️ TaskBoard 規則「@src95 不是固定操作點 ⇒ 跨模型比較只用部署 AUROC 或固定放行率」在此**不被違反**：
   OSA 把誤拒（誤殺自己人）與放行（放走 person）**同時**計入同一個數，比的是最終效用。

兩段式（GPU 前向只跑一次，門檻／π／畫圖可反覆重算）：
  --stage infer : 前向所有 checkpoint，逐樣本分數落盤 npz
  --stage plot  : 讀 npz，算曲線 + 表 + 圖（純 CPU）

複用：osdg_eval.load_backbone_diffusion（自動偵測有無 diffusion／原型）、
      joint_eval_mixed_stream.score_and_predict（0903 統一口徑的同一條前向路徑）、
      posthoc/bn_common.bn_avg（合併變異數 B 法）、test_domain_ood_scores 的 loader。
"""
import os
import sys
import csv
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE, os.path.join(HERE, "posthoc")):
    sys.path.insert(0, _p)

import numpy as np

READOUTS = ["energy", "msp", "maxlogit", "negstd", "proto", "zperp"]
PROJ = None

PACS = ["art_painting", "cartoon", "photo", "sketch"]
DEG = 57.29577951308232
N_NODES = 9

# 0903 §5 統一口徑（scripts/osdg_eval.py、ep200）的部署 AUROC 錨點——§0 自檢用。
# 出處：research/V1_baseline/0903_styleddg_baseline_4fold_results.md §5.0 / §5.1
ANCHORS = {
    ("baseline", "raw", "energy"): (0.8016, "0903 §5.1 baseline energy 原樣"),
    ("ours", "raw", "energy"): (0.8242, "0903 §5.0 重現核對 我方 energy 原樣"),
    ("ours", "raw", "proto"): (0.7956, "0903 §5.0 重現核對 我方 原型 原樣"),
    ("ours", "avg", "energy"): (0.8348, "0903 §5.0 我方 energy 平均B（舊記 0.8380 已作廢）"),
    ("ours", "avg", "proto"): (0.8117, "0903 §5.1 我方 原型 平均B（舊記 0.8145 已作廢）"),
}
ACC_ANCHORS = {("baseline", "raw"): (71.53, "0903 §3 baseline closed_acc 原樣"),
               ("ours", "avg"): (79.56, "0903 §5.1 我方 closed_acc 平均B")}


def node_source_domains(leave_out):
    """節點 i 的來源域：3 個源域各佔 3 個節點，順序同 PACS 字母序（util.py 的 virtual-node 指派）。"""
    avail = [d for d in PACS if d != leave_out]
    return [avail[min(i // 3, len(avail) - 1)] for i in range(N_NODES)]


# ─────────────────────────── stage: infer ───────────────────────────
def stage_infer(args):
    import torch
    import test_domain_ood_scores as TD
    import torch.nn.functional as F
    import util
    from osdg_eval import load_backbone_diffusion
    from joint_eval_mixed_stream import score_and_predict
    from bn_common import bn_avg, apply_bn

    device = args.device if torch.cuda.is_available() else "cpu"
    runs = {"baseline": args.baseline_desc, "ours": args.ours_desc}
    own = node_source_domains(args.leave_out)
    loaders = {d: TD.load_pacs_test_data(args.datasetRoot, d, args.batch_size, args.num_workers)[0]
               for d in PACS}
    store = {"leave_out": args.leave_out, "unknown_idx": args.unknown_idx,
             "own": np.array(own), "runs": np.array(list(runs.keys())),
             "baseline_desc": runs["baseline"], "ours_desc": runs["ours"]}

    from dood.prototype import class_centers, residual_projector, residual_score

    @torch.no_grad()
    def collect(bb, loader, centers):
        """一次前向算出全部讀出。前向路徑與 joint_eval_mixed_stream.score_and_predict 逐行相同
        （forward_to_layer3_style → forward_from_layer3），確保 energy／proto 與既有數字可對錨。
        全部定向為【高 ＝ 越像 OOD】。"""
        out = {k: [] for k in READOUTS}
        preds, labs = [], []
        for batch in loader:
            data, y, _ = util.unpack_batch(batch)
            data = data.to(device)
            z3 = bb.forward_to_layer3_style(data, communicator=None)
            logits, vec = bb.forward_from_layer3(z3)
            out["energy"].append((-torch.logsumexp(logits, 1)).cpu().numpy())
            out["msp"].append((-F.softmax(logits, 1).max(1).values).cpu().numpy())
            out["maxlogit"].append((-logits.max(1).values).cpu().numpy())
            out["negstd"].append((-logits.std(1)).cpu().numpy())      # std 大＝有明顯贏家＝像 ID
            if centers is not None:
                z = bb.project(vec)
                z = z / z.norm(dim=1, keepdim=True)                   # 單位球（project 已正規化，保險）
                cos = (z @ centers.t()).clamp(-1 + 1e-7, 1 - 1e-7)
                out["proto"].append((torch.arccos(cos).min(1).values * DEG).cpu().numpy())
                # 面讀出 ‖z⊥‖：定義同 dood.prototype.residual_score（單一真相源，
                # osdg_eval.py 的 OSCR 走同一支）
                out["zperp"].append(residual_score(z, PROJ).cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
            labs.append(np.asarray(y).flatten())
        return ({k: np.concatenate(v).astype(np.float64) for k, v in out.items() if v},
                np.concatenate(preds), np.concatenate(labs))

    for tag, desc in runs.items():
        ck_dir = os.path.join(args.exp_root, f"exp_result_{desc}")
        if not os.path.isdir(ck_dir):
            raise FileNotFoundError(f"[{tag}] 找不到 checkpoint 目錄：{ck_dir}")
        AVG = bn_avg(ck_dir, desc, N=N_NODES, ckpt_tag="final")
        for bn in ["raw", "avg"]:
            for i in range(N_NODES):
                ck = os.path.join(ck_dir, f"{desc}_node_{i}_final.pth")
                bb, dif = load_backbone_diffusion(ck, args.num_classes, device)
                if bn == "avg":
                    apply_bn(bb, AVG)
                bb.eval()
                has_proto = hasattr(bb, "prototypes")
                centers = PROJ = None
                if has_proto:
                    centers = class_centers(bb.prototypes, bb.proto_count).to(device)
                    centers = centers / centers.norm(dim=1, keepdim=True)
                    PROJ = residual_projector(centers)      # 六中心不正交 ⇒ pinv，非外積
                globals()["PROJ"] = PROJ
                for split, dom in [("src", own[i]), ("tgt", args.leave_out)]:
                    sc, pred, lab = collect(bb, loaders[dom], centers)
                    known = lab != args.unknown_idx
                    unk = ~known
                    p = f"{tag}__{bn}__{i}"
                    for k, v in sc.items():
                        if split == "src":
                            store[f"{p}__src_{k}"] = v[known].astype(np.float32)
                        else:
                            store[f"{p}__tgt_known_{k}"] = v[known].astype(np.float32)
                            store[f"{p}__tgt_unk_{k}"] = v[unk].astype(np.float32)
                    # 2026-09-11 新增 src_correct：門檻 τ 是用**全部**來源域已知樣本的 95 分位算的、
                    # 不篩分類對錯。要回答「拿來校準門檻的樣本本身分對了嗎、信心高嗎」（教授提問）
                    # 就必須有這一欄，先前只存目標域的。⚠️ 存它不改變任何既有數字。
                    # 2026-09-15 新增 pred／label：只有 0/1 對錯答不出「錯成哪一類」
                    # （sketch 的狗只有 26% ⇒ 要混淆矩陣）。label 與臂/BN 無關 ⇒ 只存一份、不隨 p 複製。
                    if split == "src":
                        store[f"{p}__src_correct"] = (pred[known] == lab[known]).astype(np.int8)
                        store[f"{p}__src_pred"] = pred[known].astype(np.int8)
                        store.setdefault(f"src_label__{i}", lab[known].astype(np.int8))
                    else:
                        store[f"{p}__tgt_known_correct"] = (pred[known] == lab[known]).astype(np.int8)
                        store[f"{p}__tgt_known_pred"] = pred[known].astype(np.int8)
                        store[f"{p}__tgt_unk_pred"] = pred[unk].astype(np.int8)
                        store.setdefault("tgt_known_label", lab[known].astype(np.int8))
                del bb, dif
                torch.cuda.empty_cache()
            print(f"  [{tag}/{bn}] 9 節點完成（讀出：{list(sc.keys())}）", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)) or ".", exist_ok=True)
    np.savez_compressed(args.out_npz, **store)
    print(f"\n落盤 → {args.out_npz}")


# ─────────────────────────── stage: plot ───────────────────────────
def _per_node(z, tag, bn, readout, q):
    """回傳逐節點 dict：門檻 τ、a_id、r_ood、A（無檢測 closed acc）、誤拒率、放行率、部署 AUROC。"""
    from sklearn.metrics import roc_auc_score
    out = []
    for i in range(N_NODES):
        p = f"{tag}__{bn}__{i}"
        if f"{p}__src_{readout}" not in z:
            return None
        s_src = z[f"{p}__src_{readout}"]
        s_k = z[f"{p}__tgt_known_{readout}"]
        s_u = z[f"{p}__tgt_unk_{readout}"]
        corr = z[f"{p}__tgt_known_correct"] > 0
        tau = float(np.quantile(s_src, q))
        accept_k = s_k <= tau
        out.append(dict(
            tau=tau,
            src_mean=float(s_src.mean()), src_std=float(s_src.std()),
            a_id=float((accept_k & corr).mean()),          # 放行且分對
            r_ood=float((s_u > tau).mean()),               # person 被拒絕（＝正確拒絕率）
            A=float(corr.mean()),                          # 無檢測的 closed acc
            fpr=float((~accept_k).mean()),                 # 誤拒率：已知類被判異常
            miss=float((s_u <= tau).mean()),               # 放行率：person 被判正常
            auroc=float(roc_auc_score(np.r_[np.zeros(len(s_k)), np.ones(len(s_u))],
                                      np.r_[s_k, s_u]))))
    return out


def _m(rows, k):
    return float(np.mean([r[k] for r in rows]))


def stage_plot(args):
    z = np.load(args.in_npz, allow_pickle=True)
    leave_out = str(z["leave_out"])
    q = args.src_quantile
    W = 100

    # 要算的所有組合（我方 energy 一併算出——同一次前向零成本，內部判讀需要完整表）
    combos = [("baseline", "raw", "energy"), ("baseline", "avg", "energy"),
              ("ours", "raw", "proto"), ("ours", "avg", "proto"),
              ("ours", "raw", "energy"), ("ours", "avg", "energy")]
    R = {}
    for c in combos:
        r = _per_node(z, *c, q)
        if r is not None:
            R[c] = r

    BNN = {"raw": "原樣", "avg": "平均B"}
    TAGN = {"baseline": "StyleDDG", "ours": "我們全套(1a-fix)"}
    RON = {"energy": "energy", "proto": "原型角距離"}
    name = lambda c: f"{TAGN[c[0]]}+{RON[c[2]]}/BN{BNN[c[1]]}"

    print("=" * W)
    print(f"§0 自檢：部署 AUROC 與 closed_acc 對 0903 報告錨點（fold={leave_out}, ep200）")
    print("=" * W)
    print(f"  {'組合':<34}{'本次':>9}{'錨點':>9}{'差':>9}   出處")
    for c, rows in R.items():
        a = ANCHORS.get(c)
        v = _m(rows, "auroc")
        if a:
            print(f"  {name(c):<34}{v:>9.4f}{a[0]:>9.4f}{v - a[0]:>+9.4f}   {a[1]}")
        else:
            print(f"  {name(c):<34}{v:>9.4f}{'—':>9}{'新測':>9}")
    print()
    for (tg, bn), (av, src) in ACC_ANCHORS.items():
        rows = R.get((tg, bn, "proto")) or R.get((tg, bn, "energy"))
        if rows:
            v = _m(rows, "A") * 100
            print(f"  closed_acc {TAGN[tg]}/BN{BNN[bn]:<4} {v:>7.2f}%  錨 {av:.2f}%  差 {v - av:+.2f}pp   {src}")
    print("  ⚠️ 差 >0.01（AUROC）或 >0.1pp（acc）要先查口徑再往下讀")

    print("\n" + "=" * W)
    print(f"§1 來源域分數與門檻：BN 平均前後（門檻 ＝ 來源域已知類別分數的 {q:.0%} 分位）")
    print("=" * W)
    print(f"  {'組合':<34}{'來源域均值':>12}{'來源域std':>11}{'門檻τ':>11}{'τ節點全距':>11}")
    for c, rows in R.items():
        taus = [r["tau"] for r in rows]
        print(f"  {name(c):<34}{_m(rows,'src_mean'):>12.4f}{_m(rows,'src_std'):>11.4f}"
              f"{np.mean(taus):>11.4f}{max(taus)-min(taus):>11.4f}")
    print("\n  ── 同一讀出、BN 平均前後的變化（★ dany 要看的「門檻變化對檢測的影響」）──")
    print(f"  {'讀出':<34}{'Δ來源域均值':>13}{'Δ門檻τ':>11}{'Δτ全距':>11}{'Δ誤拒率':>10}{'Δ放行率':>10}{'ΔAUROC':>10}")
    for tg, ro in [("baseline", "energy"), ("ours", "proto"), ("ours", "energy")]:
        a, b = R.get((tg, "raw", ro)), R.get((tg, "avg", ro))
        if not (a and b):
            continue
        ta, tb = [r["tau"] for r in a], [r["tau"] for r in b]
        print(f"  {TAGN[tg]+'+'+RON[ro]:<34}{_m(b,'src_mean')-_m(a,'src_mean'):>+13.4f}"
              f"{np.mean(tb)-np.mean(ta):>+11.4f}"
              f"{(max(tb)-min(tb))-(max(ta)-min(ta)):>+11.4f}"
              f"{_m(b,'fpr')-_m(a,'fpr'):>+10.4f}{_m(b,'miss')-_m(a,'miss'):>+10.4f}"
              f"{_m(b,'auroc')-_m(a,'auroc'):>+10.4f}")
    print("  註：τ 全距 ＝ 9 節點門檻的 max−min。全距縮小 ⇒ 各節點的判準變一致（BN 平均的直接效果）")

    print("\n" + "=" * W)
    print(f"§2 操作點（同一把刀的兩面）：誤拒率 vs 放行率")
    print("=" * W)
    print(f"  {'組合':<34}{'closed_acc':>11}{'誤拒率↓':>10}{'放行率↓':>10}{'正確拒絕':>10}{'部署AUROC↑':>12}")
    for c, rows in R.items():
        print(f"  {name(c):<34}{_m(rows,'A')*100:>10.2f}%{_m(rows,'fpr'):>10.4f}"
              f"{_m(rows,'miss'):>10.4f}{_m(rows,'r_ood'):>10.4f}{_m(rows,'auroc'):>12.4f}")

    # ── WildAcc 曲線 ──
    pis = np.linspace(0, args.pi_max, args.pi_points)
    lines, rows_csv = {}, []
    for c, rows in R.items():
        a_id, r_ood, A = _m(rows, "a_id"), _m(rows, "r_ood"), _m(rows, "A")
        lines[("det",) + c] = [(1 - p) * a_id + p * r_ood for p in pis]
        lines[("nodet", c[0], c[1])] = [(1 - p) * A for p in pis]
    for key, ys in lines.items():
        for p, y in zip(pis, ys):
            rows_csv.append(dict(leave_out=leave_out, kind=key[0], run=key[1], bn=key[2],
                                 readout=key[3] if len(key) > 3 else "",
                                 pi=round(float(p), 4), wild_acc=round(float(y), 5)))

    print("\n" + "=" * W)
    print(f"§3 ★ 開集準確率 OSA（π ＝ 混入的未知類別比例）")
    print("=" * W)
    show = [p for p in [0.0, 0.1, 0.2, 0.3, 0.5] if p <= args.pi_max + 1e-9]
    hdr = "".join(f"{'π='+format(p,'.0%'):>10}" for p in show)
    print(f"  {'方法':<40}{hdr}")
    order = [("nodet", "baseline", "raw"), ("nodet", "ours", "avg")]
    order += [("det",) + c for c in R]
    seen = set()
    for key in order:
        if key not in lines or key in seen:
            continue
        seen.add(key)
        if key[0] == "nodet":
            lab = f"{TAGN[key[1]]}（無檢測）/BN{BNN[key[2]]}"
        else:
            lab = name(key[1:])
        vals = np.interp(show, pis, lines[key]) * 100
        print(f"  {lab:<40}" + "".join(f"{v:>9.2f}%" for v in vals))

    pi0 = args.pi_report
    print(f"\n  ── π = {pi0:.0%} 的逐項拆解（dany 要的主結論）──")
    base_key = ("nodet", "baseline", "raw")
    base_v = float(np.interp(pi0, pis, lines[base_key])) * 100
    print(f"  {'方法':<40}{'OSA':>11}{'vs 無檢測baseline':>19}")
    print(f"  {'StyleDDG 無檢測（基準）':<40}{base_v:>10.2f}%{'—':>19}")
    for key in order[1:]:
        if key not in lines:
            continue
        v = float(np.interp(pi0, pis, lines[key])) * 100
        lab = (f"{TAGN[key[1]]}（無檢測）/BN{BNN[key[2]]}" if key[0] == "nodet" else name(key[1:]))
        print(f"  {lab:<40}{v:>10.2f}%{v - base_v:>+18.2f}pp")

    # ── 圖 ──
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK TC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(7.2, 5))
        STY = {("baseline", "raw", "energy"): ("tab:blue", "-", "s"),
               ("baseline", "avg", "energy"): ("tab:blue", "--", "^"),
               ("ours", "raw", "proto"): ("tab:green", "-", "o"),
               ("ours", "avg", "proto"): ("tab:green", "--", "D")}
        ax.plot(pis, np.array(lines[base_key]) * 100, "-", color="crimson", lw=2.4,
                marker="x", label="StyleDDG no detector")
        if ("nodet", "ours", "avg") in lines:
            ax.plot(pis, np.array(lines[("nodet", "ours", "avg")]) * 100, ":", color="dimgray",
                    lw=1.6, label="ours no detector (classifier only)")
        for c, (col, ls, mk) in STY.items():
            if ("det",) + c in lines:
                lab = f"{'StyleDDG+energy' if c[0]=='baseline' else 'ours+prototype'} / BN {'raw' if c[1]=='raw' else 'avg'}"
                ax.plot(pis, np.array(lines[("det",) + c]) * 100, ls, color=col, marker=mk,
                        ms=4, label=lab)
        ax.axvline(pi0, color="gray", lw=0.8, alpha=.6)
        ax.set_xlabel(r"unknown-class ratio $\pi$ in the test stream")
        ax.set_ylabel("deployment decision accuracy (%)")
        ax.set_title(f"{leave_out} (unseen) — deployment decision accuracy, threshold @ source-{q:.0%}")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(args.out_dir, f"dda_{leave_out}.png")
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f"\n  圖 → {png}")
    except Exception as e:
        print(f"\n  （畫圖略過：{e}）")

    csv_path = os.path.join(args.out_dir, f"dda_{leave_out}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"  表 → {csv_path}")

    node_csv = os.path.join(args.out_dir, f"per_node_{leave_out}.csv")
    with open(node_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "bn", "readout", "node", "tau", "src_mean", "src_std",
                    "a_id", "r_ood", "A", "fpr", "miss", "auroc"])
        for c, rows in R.items():
            for i, r in enumerate(rows):
                w.writerow(list(c) + [i] + [round(r[k], 6) for k in
                           ["tau", "src_mean", "src_std", "a_id", "r_ood", "A", "fpr", "miss", "auroc"]])
    print(f"  逐節點 → {node_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["infer", "plot"])
    p.add_argument("--leave_out", default="cartoon", choices=PACS)
    p.add_argument("--unknown_idx", type=int, default=6)
    p.add_argument("--num_classes", type=int, default=6)
    # infer
    p.add_argument("--exp_root", default=".")
    p.add_argument("--baseline_desc",
                   default="v1_stage2_leave_cartoon_nodiff_osdg_excl_person_seed2026_topo1234")
    p.add_argument("--ours_desc",
                   default="v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_"
                           "aggbn_osdg_excl_person_seed2026_topo1234_fix")
    p.add_argument("--datasetRoot", default="../datasets/")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_npz", default="results/osa/cartoon.npz")
    # plot
    p.add_argument("--in_npz", default="results/osa/cartoon.npz")
    p.add_argument("--out_dir", default="results/osa")
    p.add_argument("--src_quantile", type=float, default=0.95)
    p.add_argument("--pi_max", type=float, default=0.5)
    p.add_argument("--pi_points", type=int, default=51)
    p.add_argument("--pi_report", type=float, default=0.2)
    a = p.parse_args()
    stage_infer(a) if a.stage == "infer" else stage_plot(a)


if __name__ == "__main__":
    main()
