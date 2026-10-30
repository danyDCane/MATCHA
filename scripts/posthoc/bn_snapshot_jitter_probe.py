"""BN 快照雜訊探針（post-hoc、零重訓、不覆寫 checkpoint）。

問題（1027）：同一份程式只因換卡（最後一兩位數不同）重跑，sketch 的 ep200 面 OSA −2.54、泛化 −0.99。
這個差有多少是「最後一刻 BN 統計量的快照」造成的（程式改得動），多少是「兩支走到不同的權重」（改不動）？

依據：學習率照餘弦降到 0，最後 10 個 epoch 權重幾乎不動，但同一支 run 的逐 epoch 目標域準確率仍晃
（sketch 標準差 0.37）。BN running 統計量是最近約 20 批（momentum 0.1）的移動平均，學習率為 0 也照樣在變。

條件（每組 checkpoint、epoch_200、九台）：
  orig   ＝checkpoint 自帶 BN → 平均B（＝成績表口徑；必須重現登記值，當錨點）
  snap_r ＝從 checkpoint BN 出發、momentum 0.1、再吃 20 批隨機抽的來源域乾淨圖 → 平均B
           （模擬「停在另一個時刻」的快照；r=1..5）
  full   ＝BN 歸零、momentum=None 累積平均、吃整份來源域乾淨圖 → 平均B（候選改法：確定性重估）
  half_s ＝同 full 但只用隨機一半的圖（s=1..3；看候選改法本身穩不穩）

⚠️ snap_r 用乾淨圖，訓練時是 98% 加風格的圖 ⇒ 與訓練時的快照**不是同一個量**；
   它偏小不代表訓練時的快照雜訊小（單向證據）。full 就是候選改法本身 ⇒ 判讀以 full 為準。
⚠️ BN 重估只用來源域（不碰目標域）；person 類別排除（訓練時也排除）。
   前向路徑＝forward_to_layer3_style(communicator=None) → forward_from_layer3（與 open_set_accuracy 相同；
   model.eval() 關掉風格擾動，只把 BN 子模組設 train 以更新統計量）。

事前判準（結果出來前寫死）：
  sketch 登記 vs 重跑的面 OSA 差 2.54。若 ①full 後兩支的差 ≤1.27（縮一半以上），或
  ②snap_r 五次的全距 ≥1.27 ⇒ BN 快照是主要來源之一、值得改程式；兩者皆否 ⇒ 差在權重、BN 改法無效。

用法：
  venv_matcha/bin/python scripts/posthoc/bn_snapshot_jitter_probe.py --out_json <path>
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn

import util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from open_set_accuracy import node_source_domains
from dood.prototype import class_centers, residual_projector, residual_score

N = 9
UNK = 6
WT = "/home/server5090/Desktop/M11307320/MATCHA/.claude/worktrees"
SETS = [  # (標籤, leave_out, checkpoint 目錄, desc)
    ("sketch_reg", "sketch", f"{ROOT}/exp_result_v1_stage2_leave_sketch_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234",
     "v1_stage2_leave_sketch_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"),
    ("sketch_new", "sketch", f"{WT}/repro_59865aa/exp_result_v1_stage2_leave_sketch_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234",
     "v1_stage2_leave_sketch_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"),
    ("cartoon_reg", "cartoon", f"{ROOT}/exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix",
     "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"),
    ("cartoon_new", "cartoon", f"{WT}/repro_5c6556d/exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix",
     "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"),
    ("cartoon_verD", "cartoon", f"{ROOT}/exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234_fix",
     "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234_fix"),
]


def bn_modules(m):
    return [x for x in m.modules() if isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]


def bn_state(m):
    return {k: v.detach().clone().float() for k, v in m.state_dict().items()
            if k.endswith("running_mean") or k.endswith("running_var")}


def merged_avg(states):
    """平均B：mean(var_i + mean_i²) − mean(mean_i)²（同 bn_common.bn_avg）。"""
    A = {}
    for k in states[0]:
        if k.endswith("running_mean"):
            A[k] = torch.stack([s[k] for s in states]).mean(0)
    for k in states[0]:
        if k.endswith("running_var"):
            mk = k.replace("running_var", "running_mean")
            mi = torch.stack([s[mk] for s in states]); vi = torch.stack([s[k] for s in states])
            A[k] = (vi + mi ** 2).mean(0) - mi.mean(0) ** 2
    return A


def load_bn(m, st):
    sd = m.state_dict()
    for k, v in st.items():
        sd[k].copy_(v.to(sd[k].device).to(sd[k].dtype))


def cache_domain(root, dom, bs, nw):
    """整個 domain 一次解碼成 CPU float32 張量（測試轉換、不打亂），之後每個條件重複用。"""
    loader = TD.load_pacs_test_data(root, dom, bs, nw)[0]
    xs, ys = [], []
    for b in loader:
        x, y, _ = util.unpack_batch(b)
        xs.append(x); ys.append(torch.as_tensor(np.asarray(y)).flatten())
    return torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def refresh_bn(m, X, idx, bs, dev, momentum, reset):
    """凍權重，只更新 BN running 統計量。"""
    m.eval()
    bns = bn_modules(m)
    for b in bns:
        if reset:
            b.reset_running_stats()
        b.momentum = momentum
        b.train()
    for s in range(0, len(idx), bs):
        x = X[idx[s:s + bs]].to(dev, non_blocking=True)
        m.forward_from_layer3(m.forward_to_layer3_style(x, communicator=None))
    for b in bns:
        b.momentum = 0.1
        b.eval()


@torch.no_grad()
def scores(m, X, bs, dev, proj):
    zs, es, ps = [], [], []
    for s in range(0, len(X), bs):
        x = X[s:s + bs].to(dev, non_blocking=True)
        logits, vec = m.forward_from_layer3(m.forward_to_layer3_style(x, communicator=None))
        z = m.project(vec); z = z / z.norm(dim=1, keepdim=True)
        zs.append(residual_score(z, proj).cpu()); es.append((-torch.logsumexp(logits, 1)).cpu())
        ps.append(logits.argmax(1).cpu())
    return torch.cat(zs).double().numpy(), torch.cat(es).double().numpy(), torch.cat(ps).numpy()


def evaluate(models, projs, own, data, leave_out, bs, dev):
    """九台各自用（已塞好的）BN 算：泛化、面 AUROC、面 OSA(20%)、誤拒率；回傳九台平均與逐台。"""
    from sklearn.metrics import roc_auc_score
    Xt, yt = data[leave_out]; known_t = (yt != UNK).numpy()
    rows = []
    for i, m in enumerate(models):
        Xs, ys = data[own[i]]; known_s = (ys != UNK).numpy()
        s_src, _, _ = scores(m, Xs[known_s], bs, dev, projs[i])
        s_t, e_t, p_t = scores(m, Xt, bs, dev, projs[i])
        corr = p_t[known_t] == yt.numpy()[known_t]
        tau = float(np.quantile(s_src, 0.95))
        acc_k = s_t[known_t] <= tau
        a_id = float((acc_k & corr).mean()); r_ood = float((s_t[~known_t] > tau).mean())
        lab = np.r_[np.zeros(known_t.sum()), np.ones((~known_t).sum())]
        rows.append(dict(acc=100 * float(corr.mean()),
                         auroc=float(roc_auc_score(lab, np.r_[s_t[known_t], s_t[~known_t]])),
                         osa=100 * (0.8 * a_id + 0.2 * r_ood), fpr=100 * float((~acc_k).mean()),
                         tau=tau, src_mean=float(s_src.mean())))
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return mean, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasetRoot", default="../datasets/")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--snap_batches", type=int, default=20)
    ap.add_argument("--n_snap", type=int, default=5)
    ap.add_argument("--n_half", type=int, default=3)
    ap.add_argument("--sets", default="all")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    dev = "cuda"
    sets = SETS if args.sets == "all" else [s for s in SETS if s[0] in args.sets.split(",")]
    out = json.load(open(args.out_json)) if os.path.exists(args.out_json) else {}
    data_fold, data = None, None
    for tag, lo, ck, desc in sets:
        if tag in out:
            print(f"[skip] {tag} 已有結果"); continue
        if data_fold != lo:
            data = {d: cache_domain(args.datasetRoot, d, args.bs, 4) for d in ["art_painting", "cartoon", "photo", "sketch"]}
            data_fold = lo
            print(f"[data] {lo}: " + ", ".join(f"{d}={len(v[1])}" for d, v in data.items()), flush=True)
        own = node_source_domains(lo)
        models, projs, ck_bn = [], [], []
        for i in range(N):
            m, _dif = load_backbone_diffusion(f"{ck}/{desc}_node_{i}_epoch_200.pth", 6, dev)
            del _dif
            c = class_centers(m.prototypes, m.proto_count).to(dev); c = c / c.norm(dim=1, keepdim=True)
            models.append(m); projs.append(residual_projector(c)); ck_bn.append(bn_state(m))
        src_idx = {d: torch.nonzero(data[d][1] != UNK).flatten() for d in set(own)}
        res = {}

        def run(cond, per_node_bn):
            A = merged_avg(per_node_bn)
            for m in models:
                load_bn(m, A)
            mean, rows = evaluate(models, projs, own, data, lo, args.bs, dev)
            res[cond] = dict(mean=mean, nodes=rows)
            print(f"  {tag:13} {cond:8} acc={mean['acc']:.2f} auroc={mean['auroc']:.4f} "
                  f"osa={mean['osa']:.2f} fpr={mean['fpr']:.2f} tau={mean['tau']:.4f}", flush=True)

        run("orig", ck_bn)
        for r in range(1, args.n_snap + 1):
            g = torch.Generator().manual_seed(1000 + r)
            st = []
            for i, m in enumerate(models):
                load_bn(m, ck_bn[i])
                pool = src_idx[own[i]]
                idx = pool[torch.randint(len(pool), (args.snap_batches * args.bs,), generator=g)]
                refresh_bn(m, data[own[i]][0], idx, args.bs, dev, momentum=0.1, reset=False)
                st.append(bn_state(m))
            run(f"snap{r}", st)
        # ⚠️ 必須打亂順序：資料集依類別排序，照原順序每批幾乎只有一類；PyTorch 的累積平均
        #    （momentum=None）是「各批批內變異數」的平均 ⇒ 單類批會把變異數低估（1027 首跑踩到）。
        g = torch.Generator().manual_seed(3000)
        st = []
        for i, m in enumerate(models):
            pool = src_idx[own[i]]
            refresh_bn(m, data[own[i]][0], pool[torch.randperm(len(pool), generator=g)], args.bs, dev,
                       momentum=None, reset=True)
            st.append(bn_state(m))
        run("full", st)
        for s in range(1, args.n_half + 1):
            g = torch.Generator().manual_seed(2000 + s)
            st = []
            for i, m in enumerate(models):
                pool = src_idx[own[i]]
                idx = pool[torch.randperm(len(pool), generator=g)[: len(pool) // 2]]
                refresh_bn(m, data[own[i]][0], idx, args.bs, dev, momentum=None, reset=True)
                st.append(bn_state(m))
            run(f"half{s}", st)
        out[tag] = res
        json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=1)
        del models
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
