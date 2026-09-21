"""0831 AdaIN 介入形式對齊：複驗 rel_loss 的 no-go（plan: research/prototype_probe/0831_adain_form_alignment_plan.md）

問題：0829 判死 rel_loss 用的射程 6.8% 是「域平均一刀切、ε=0」量的，
      而訓練時 StyleShift 走的是 batch 級統計量 + DSU 採樣 ⇒ 介入形式與訓練形式不一致。

四形式單變因遞進（隔離三個變數）：
  3a  域級 + 0829 舊算法(std unbiased=True)          ← 重現 0829 的 0.0080 / 6.8% / 0.8358
  3a' 域級 + 訓練算法(compute_layer_style_stats)      ← 隔離「std 算法」變數，預期 ≈0
  3b  batch 級(64 張) + 中心點(ε=0)                   ← 隔離「batch vs 域」變數
  3c  batch 級 + DSU 採樣(ε~N(0,1))                   ← 完全對齊訓練形式（generate_target_style_from_neighbor）

判準（plan §7，事前寫死）：三關全過才算重開 no-go
  R2 控制B < 0.01（0829 是 +0.0137 沒過）｜R3 ② 降幅 ≥0.030（26%）｜R4 部署 AUROC > 0.8380
附帶：分類準確率（補現架構版 P3；0829 把 forward_from_layer3 的 logits 丟掉了）
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "."); sys.path.insert(0, "scripts/posthoc")
import torch, util, csv
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
from style_transforms import adain, generate_target_style_from_neighbor
from style_stats import compute_layer_style_stats
from bn_common import bn_avg, apply_bn
from scipy.stats import rankdata
from torch.utils.data import Subset, DataLoader

OUT = "research/outputs/0831_adain_form"; os.makedirs(OUT, exist_ok=True); ROWS = []
PACS = ["art_painting", "cartoon", "photo", "sketch"]
DESC = "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
CK = "exp_result_" + DESC; leave = "cartoon"; UNK = 6; N = 9; DEG = 57.29577951308232
LAYERS = ("layer1", "layer2", "layer3")
M = int(os.environ.get("M", "3"))                       # DSU/batch 重抽次數
NODES = int(os.environ.get("NODES", str(N)))            # smoke 用
BASES = os.environ.get("BASES", "bn,nobn").split(",")   # smoke 用
BATCH = 64                                              # train.py:2242 --bs 預設 64

avail = [d for d in PACS if d != leave]; per = N // len(avail)
tgt = TD.load_pacs_test_data("../datasets/", leave, 64, 4)[0]
src = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in avail}


def auroc(pos, neg):
    a = np.concatenate([pos, neg]); r = rankdata(a, method="average")
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------- 統計量生成：四形式
@torch.no_grad()
def stats_domain_old(bb, ld):
    """形式 3a ＝ 0829 dom_stats 逐字複製（std unbiased=True、無 eta）。重現用，勿改。"""
    acc = {k: [[], []] for k in LAYERS}
    for b in ld:
        d, _, _ = util.unpack_batch(b); d = d.to("cuda")
        Fm = bb.extract_features_to_layer3(d)
        for k in acc:
            f = Fm[k]; B, C, H, W = f.shape; fl = f.view(B, C, -1)
            acc[k][0].append(fl.mean(2).cpu()); acc[k][1].append(fl.std(2).cpu())
    return {k: (torch.cat(v[0]).mean(0).cuda(), torch.cat(v[1]).mean(0).cuda()) for k, v in acc.items()}


@torch.no_grad()
def stats_domain_train(bb, ld):
    """形式 3a' ＝ 整個域當一個大 batch，用訓練的 compute_layer_style_stats 公式（樣本數加權）。"""
    s = {k: [None, None] for k in LAYERS}; n = 0
    for b in ld:
        d, _, _ = util.unpack_batch(b); d = d.to("cuda"); bs = d.size(0); n += bs
        Fm = bb.extract_features_to_layer3(d)
        for k in LAYERS:
            st = compute_layer_style_stats(Fm[k])
            mu, sg = st["mu_bar"] * bs, st["sigma_bar"] * bs
            s[k][0] = mu if s[k][0] is None else s[k][0] + mu
            s[k][1] = sg if s[k][1] is None else s[k][1] + sg
    return {k: (s[k][0] / n, s[k][1] / n) for k in LAYERS}


@torch.no_grad()
def stats_batch(bb, ld, idx, dsu, gen):
    """形式 3b/3c ＝ 抽 BATCH 張（一個訓練 batch），用訓練同函式。dsu=True 時再過 DSU 採樣。"""
    sub = DataLoader(Subset(ld.dataset, idx), batch_size=BATCH, shuffle=False, num_workers=0)
    b = next(iter(sub)); d, _, _ = util.unpack_batch(b); d = d.to("cuda")
    Fm = bb.extract_features_to_layer3(d)
    out = {}
    for k in LAYERS:
        st = compute_layer_style_stats(Fm[k])
        if dsu:
            # 訓練同函式：beta = mu_bar + eps*sqrt(Sigma_mu_sq), gamma = sigma_bar + eps*sqrt(Sigma_sigma_sq)
            torch.manual_seed(int(gen.integers(0, 2**31 - 1)))
            out[k] = generate_target_style_from_neighbor({k: st}, k, torch.device("cuda"))
        else:
            out[k] = (st["mu_bar"], st["sigma_bar"])
    return out


def shuffle_stats(st, seed):
    g = torch.Generator().manual_seed(seed)
    return {k: (v[0][torch.randperm(len(v[0]), generator=g).to(v[0].device)],
                v[1][torch.randperm(len(v[1]), generator=g).to(v[1].device)]) for k, v in st.items()}


# ---------------------------------------------------------------- 量測（＋分類準確率）
@torch.no_grad()
def collect(bb, ld, C, U, st=None):
    """回傳 (到自己類別角度, ||z_perp||, y, 分類正確與否)"""
    ow, zp, ys, ok = [], [], [], []
    for b in ld:
        d, y, _ = util.unpack_batch(b); d = d.to("cuda"); y = np.asarray(y).flatten()
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(d))))
        h = bb.backbone.layer1(h)
        if st: h = adain(h, *st["layer1"])
        h = bb.backbone.layer2(h)
        if st: h = adain(h, *st["layer2"])
        h = bb.backbone.layer3(h)
        if st: h = adain(h, *st["layer3"])
        logits, vec = bb.forward_from_layer3(h)          # ★ 0829 把 logits 丟掉了，這裡接住
        z = bb.project(vec); z = z / z.norm(dim=1, keepdim=True)
        zn = z.cpu().numpy().astype(np.float64)
        zp.append(np.sqrt(np.maximum(1 - ((zn @ U) ** 2).sum(1), 0)))
        a = torch.arccos((z @ C.t()).clamp(-1 + 1e-7, 1 - 1e-7)).cpu().numpy() * DEG
        m = y != UNK
        o = np.full(len(y), np.nan); o[m] = a[m][np.arange(m.sum()), y[m].astype(int)]
        pred = logits.argmax(1).cpu().numpy()
        c = np.full(len(y), np.nan); c[m] = (pred[m] == y[m]).astype(float)
        ow.append(o); ys.append(y); ok.append(c)
    return np.concatenate(ow), np.concatenate(zp), np.concatenate(ys), np.concatenate(ok)


def run_arm(bb, ld, C, U, st_list):
    """對 st_list 中每一組統計量各量一次，回傳各指標的平均（M 次重抽用）。"""
    acc = {k: [] for k in ("own", "zp2", "sd2", "zp3", "sd3", "auc", "cls")}
    for st in st_list:
        o, zp, y, ok = collect(bb, ld, C, U, st)
        m2, m3 = y != UNK, y == UNK
        acc["own"].append(np.nanmean(o)); acc["zp2"].append(zp[m2].mean())
        acc["sd2"].append(zp[m2].std(ddof=1)); acc["cls"].append(np.nanmean(ok))
        if m3.sum() > 0:
            acc["zp3"].append(zp[m3].mean()); acc["sd3"].append(zp[m3].std(ddof=1))
            acc["auc"].append(auroc(zp[m3], zp[m2]))
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}, int(m3.sum())


# ---------------------------------------------------------------- 主流程
for BN in [b == "bn" for b in BASES]:
    tag = "BN 平均 B（現行協定、判決用）" if BN else "不做 BN 平均（重現 0818 自檢用）"
    print("\n" + "#" * 116); print(f"###  基底：{tag}   M={M}  NODES={NODES}"); print("#" * 116, flush=True)
    AVG = bn_avg(CK, DESC, N=N) if BN else None
    R = {}
    for i in range(NODES):
        bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
        if BN: apply_bn(bb, AVG)
        bb.eval()
        C = class_centers(bb.prototypes, bb.proto_count)
        Cn = (C / C.norm(dim=1, keepdim=True)).cpu().numpy().astype(np.float64)
        U = np.linalg.svd(Cn.T, full_matrices=False)[0]
        own_d = avail[min(i // per, len(avail) - 1)]
        rng = np.random.default_rng(2026 + i)

        # 四形式的來源域統計量
        st_a  = stats_domain_old(bb, src[own_d])                       # 3a  域級+舊算法
        st_ap = stats_domain_train(bb, src[own_d])                     # 3a' 域級+訓練算法
        n_src = len(src[own_d].dataset)
        idxs  = [rng.choice(n_src, BATCH, replace=False) for _ in range(M)]
        st_b  = [stats_batch(bb, src[own_d], ix, False, rng) for ix in idxs]   # 3b  batch+中心點
        st_c  = [stats_batch(bb, src[own_d], ix, True,  rng) for ix in idxs]   # 3c  batch+DSU
        # 控制 A（cartoon 自己的統計量）與零點（打亂 channel）
        st_ctl_a = stats_domain_old(bb, tgt)
        n_tgt = len(tgt.dataset)
        st_ctl_c = [stats_batch(bb, tgt, rng.choice(n_tgt, BATCH, replace=False), True, rng)]
        st_rnd_a = shuffle_stats(st_a, 2026 + i)
        st_rnd_c = [shuffle_stats(st_c[0], 3026 + i)]

        ARMS = [
            ("1.①來源域(下界)",              src[own_d], [None]),
            ("2.②③cartoon原樣(基準)",        tgt,        [None]),
            ("3a.★主臂 域級+舊算法",          tgt,        [st_a]),
            ("3a'.★主臂 域級+訓練算法",       tgt,        [st_ap]),
            ("3b.★主臂 batch級+中心點",       tgt,        st_b),
            ("3c.★主臂 batch級+DSU",          tgt,        st_c),
            ("4a.控制A cartoon自己 域級",     tgt,        [st_ctl_a]),
            ("4c.控制A cartoon自己 batch+DSU",tgt,        st_ctl_c),
            ("5a.控制B ①+自己 域級舊算法",    src[own_d], [st_a]),
            ("5a'.控制B ①+自己 域級訓練算法", src[own_d], [st_ap]),
            ("5b.控制B ①+自己 batch級",       src[own_d], st_b),
            ("5c.控制B ①+自己 batch+DSU",     src[own_d], st_c),
            ("6a.零點 隨機 域級",             tgt,        [st_rnd_a]),
            ("6c.零點 隨機 batch+DSU",        tgt,        st_rnd_c),
        ]
        for name, ld, stl in ARMS:
            R.setdefault(name, {k: [] for k in ("own", "zp2", "sd2", "zp3", "sd3", "auc", "cls")})
            v, n3 = run_arm(bb, ld, C, U, stl)
            for k in R[name]: R[name][k].append(v[k])
            ROWS.append(dict(basis="bn_avg_B" if BN else "no_bn_avg", arm=name, node=i, own_domain=own_d,
                             n_stats=len(stl), angle_own_deg=round(v["own"], 3),
                             zperp2_mean=round(v["zp2"], 4), zperp2_std=round(v["sd2"], 4),
                             zperp3_mean=("" if np.isnan(v["zp3"]) else round(v["zp3"], 4)),
                             zperp3_std=("" if np.isnan(v["sd3"]) else round(v["sd3"], 4)),
                             deploy_auroc=("" if np.isnan(v["auc"]) else round(v["auc"], 4)),
                             cls_acc=round(v["cls"], 4)))
        del bb; torch.cuda.empty_cache()
        print(f"  node_{i} 完成（own={own_d}）", flush=True)

    # ---- 彙整
    mv = lambda k, f: (np.mean(R[k][f]) if R[k][f] else float("nan"))
    print("\n" + "=" * 116)
    print(f"{'臂':<34}{'到自己類別°':>12}{'②‖z⊥‖':>10}{'②std':>8}{'③‖z⊥‖':>10}{'③std':>8}{'部署AUROC':>11}{'分類acc':>9}")
    print("-" * 116)
    for k in R:
        print(f"{k:<34}{mv(k,'own'):12.2f}{mv(k,'zp2'):10.4f}{mv(k,'sd2'):8.4f}"
              f"{mv(k,'zp3'):10.4f}{mv(k,'sd3'):8.4f}{mv(k,'auc'):11.4f}{mv(k,'cls'):9.4f}")
    print("-" * 116)
    lo, base = mv("1.①來源域(下界)", "zp2"), mv("2.②③cartoon原樣(基準)", "zp2"); tot = base - lo
    b3, bauc = mv("2.②③cartoon原樣(基準)", "zp3"), mv("2.②③cartoon原樣(基準)", "auc")
    print(f"  可解釋總量（②基準 {base:.4f} − ①下界 {lo:.4f}）= {tot:.4f}")
    for k in ["3a.★主臂 域級+舊算法", "3a'.★主臂 域級+訓練算法", "3b.★主臂 batch級+中心點",
              "3c.★主臂 batch級+DSU", "4a.控制A cartoon自己 域級", "6a.零點 隨機 域級"]:
        print(f"  {k:<30} ② 降 {base-mv(k,'zp2'):+.4f} ⇒ 解釋 {(base-mv(k,'zp2'))/tot*100:6.1f}%"
              f"   ③ 變化 {mv(k,'zp3')-b3:+.4f}   AUROC {mv(k,'auc'):.4f}   acc {mv(k,'cls'):.4f}")
    print(f"\n  ── R2 控制B（①no-op，門檻 <0.01）──  ①原樣 {lo:.4f}")
    for k in ["5a.控制B ①+自己 域級舊算法", "5a'.控制B ①+自己 域級訓練算法",
              "5b.控制B ①+自己 batch級", "5c.控制B ①+自己 batch+DSU"]:
        d = mv(k, "zp2") - lo
        print(f"    {k:<32} ‖z⊥‖ {mv(k,'zp2'):.4f}   Δ {d:+.4f}   {'✅ 過' if abs(d) < 0.01 else '❌ 沒過'}")
    print(f"\n  ── R6 變數分離 ──")
    a, ap = mv("3a.★主臂 域級+舊算法", "zp2"), mv("3a'.★主臂 域級+訓練算法", "zp2")
    b, c = mv("3b.★主臂 batch級+中心點", "zp2"), mv("3c.★主臂 batch級+DSU", "zp2")
    print(f"    std 算法效應 (3a'−3a) = {a-ap:+.4f}   {'✅ 可忽略' if abs(a-ap) < 0.002 else '⚠️ 不可忽略'}")
    print(f"    batch vs 域 (3b−3a')  = {ap-b:+.4f}")
    print(f"    DSU 採樣    (3c−3b)   = {b-c:+.4f}")
    print(f"\n  ── 三關（plan §7；全過才算重開 no-go）──")
    d2c = base - c
    print(f"    R2 控制B(3c) <0.01      : {mv('5c.控制B ①+自己 batch+DSU','zp2')-lo:+.4f}  "
          f"{'✅' if abs(mv('5c.控制B ①+自己 batch+DSU','zp2')-lo) < 0.01 else '❌'}")
    print(f"    R3 ② 降幅 ≥0.030 (26%)  : {d2c:+.4f} ({d2c/tot*100:.1f}%)  {'✅' if d2c >= 0.030 else '❌'}")
    print(f"    R4 部署 AUROC >0.8380   : {mv('3c.★主臂 batch級+DSU','auc'):.4f}  "
          f"{'✅' if mv('3c.★主臂 batch級+DSU','auc') > 0.8380 else '❌'}")
    if not BN:
        print(f"\n  ★ 0818 對照（角距離）：①下界 {mv('1.①來源域(下界)','own'):.2f}° (報 29.63) ｜"
              f" ②基準 {mv('2.②③cartoon原樣(基準)','own'):.2f}° (報 53.57) ｜"
              f" 3a 主臂 {mv('3a.★主臂 域級+舊算法','own'):.2f}° (報 48.48)")
    print("=" * 116, flush=True)

TAG="_".join(BASES)
with open(f"{OUT}/per_node_{TAG}.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(ROWS[0])); w.writeheader(); w.writerows(ROWS)
print(f"\nCSV: {OUT}/per_node_{TAG}.csv（{len(ROWS)} 列）")
