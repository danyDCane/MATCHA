"""dany 2026-08-29 核准的驗證：channel μ/σ 這個軸能解釋 ‖z⊥‖ 污染的多少？
＝ rel_loss 的射程上界（必要非充分：救不了就一定死，救得了還要過「外推到未見畫風」那關）。
基於 0818 adain_intervention_angle.py，四項修改：加 BN 平均／加 ‖z⊥‖／**納入 person**／加部署 AUROC。
判準（事前寫死，dany 核准）：
  主判準 ② 的 ‖z⊥‖ 降幅（總空間 0.6289−0.5116=0.1173）：≥50% 值得發車｜26–50% 邊際｜<26% 不跑
  副判準 ③ person：|Δ|<0.01 ⇒ 選擇性成立｜降 >0.05 ⇒ 一視同仁 ⇒ 直接死
  健全性：控制組A <主臂1/3｜控制組B <0.01｜不平均基底須重現 0818 的 53.57→48.48°
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "."); sys.path.insert(0, "scripts/posthoc")
import torch, util, csv
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
from style_transforms import adain
from bn_common import bn_avg, apply_bn
from scipy.stats import rankdata
OUT="research/outputs/0829_adain"; os.makedirs(OUT, exist_ok=True); ROWS=[]

PACS = ["art_painting", "cartoon", "photo", "sketch"]
DESC = "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
CK = "exp_result_" + DESC; leave = "cartoon"; UNK = 6; N = 9; DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]; per = N // len(avail)
tgt = TD.load_pacs_test_data("../datasets/", leave, 64, 4)[0]
src = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in avail}
def auroc(pos, neg):
    a = np.concatenate([pos, neg]); r = rankdata(a, method="average")
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

@torch.no_grad()
def dom_stats(bb, ld):
    acc = {k: [[], []] for k in ("layer1", "layer2", "layer3")}
    for b in ld:
        d, _, _ = util.unpack_batch(b); d = d.to("cuda")
        Fm = bb.extract_features_to_layer3(d)
        for k in acc:
            f = Fm[k]; B, C, H, W = f.shape; fl = f.view(B, C, -1)
            acc[k][0].append(fl.mean(2).cpu()); acc[k][1].append(fl.std(2).cpu())
    return {k: (torch.cat(v[0]).mean(0).cuda(), torch.cat(v[1]).mean(0).cuda()) for k, v in acc.items()}

@torch.no_grad()
def collect(bb, ld, C, U, st=None):
    """回傳 (到自己類別角度, ‖z⊥‖, y)"""
    ow, zp, ys = [], [], []
    for b in ld:
        d, y, _ = util.unpack_batch(b); d = d.to("cuda"); y = np.asarray(y).flatten()
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(d))))
        h = bb.backbone.layer1(h)
        if st: h = adain(h, *st["layer1"])
        h = bb.backbone.layer2(h)
        if st: h = adain(h, *st["layer2"])
        h = bb.backbone.layer3(h)
        if st: h = adain(h, *st["layer3"])
        _, vec = bb.forward_from_layer3(h)
        z = bb.project(vec); z = z / z.norm(dim=1, keepdim=True)
        zn = z.cpu().numpy().astype(np.float64)
        zp.append(np.sqrt(np.maximum(1 - ((zn @ U)**2).sum(1), 0)))
        a = torch.arccos((z @ C.t()).clamp(-1+1e-7, 1-1e-7)).cpu().numpy() * DEG
        m = y != UNK
        o = np.full(len(y), np.nan); o[m] = a[m][np.arange(m.sum()), y[m].astype(int)]
        ow.append(o); ys.append(y)
    return np.concatenate(ow), np.concatenate(zp), np.concatenate(ys)

for BN in [True, False]:
    tag = "BN 平均 B（現行協定、判決用）" if BN else "不做 BN 平均（重現 0818 自檢用）"
    print("\n" + "#"*104); print(f"###  基底：{tag}"); print("#"*104, flush=True)
    AVG = bn_avg(CK, DESC, N=N) if BN else None
    ARMS = ["1.①來源域(下界)", "2.②③cartoon 原樣(基準)", "3.★②③+來源域統計量(主臂)",
            "4.控制A ②③+cartoon自己統計量", "5.控制B ①+自己統計量", "6.零點 ②③+隨機統計量"]
    R = {k: {"own": [], "zp2": [], "zp3": [], "sd2": [], "sd3": [], "auc": []} for k in ARMS}
    rng = np.random.default_rng(2026)
    for i in range(N):
        bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
        if BN: apply_bn(bb, AVG)
        bb.eval()
        C = class_centers(bb.prototypes, bb.proto_count)
        Cn = (C / C.norm(dim=1, keepdim=True)).cpu().numpy().astype(np.float64)
        U = np.linalg.svd(Cn.T, full_matrices=False)[0]
        own_d = avail[min(i // per, len(avail) - 1)]
        st_src, st_ctl = dom_stats(bb, src[own_d]), dom_stats(bb, tgt)
        g = torch.Generator().manual_seed(2026 + i)
        st_rnd = {k: (v[0][torch.randperm(len(v[0]), generator=g).to(v[0].device)],
                      v[1][torch.randperm(len(v[1]), generator=g).to(v[1].device)])
                  for k, v in st_src.items()}
        for k, (ld, st) in {ARMS[0]: (src[own_d], None), ARMS[1]: (tgt, None), ARMS[2]: (tgt, st_src),
                            ARMS[3]: (tgt, st_ctl), ARMS[4]: (src[own_d], st_src), ARMS[5]: (tgt, st_rnd)}.items():
            o, zp, y = collect(bb, ld, C, U, st); m2, m3 = y != UNK, y == UNK
            R[k]["own"].append(np.nanmean(o)); R[k]["zp2"].append(zp[m2].mean()); R[k]["sd2"].append(zp[m2].std(ddof=1))
            row = dict(basis="bn_avg_B" if BN else "no_bn_avg", arm=k, node=i, own_domain=own_d,
                       angle_own_deg=round(float(np.nanmean(o)), 3), n2=int(m2.sum()), n3=int(m3.sum()),
                       zperp2_mean=round(float(zp[m2].mean()), 4), zperp2_std=round(float(zp[m2].std(ddof=1)), 4),
                       zperp3_mean="", zperp3_std="", deploy_auroc="")
            if m3.sum() > 0:
                R[k]["zp3"].append(zp[m3].mean()); R[k]["sd3"].append(zp[m3].std(ddof=1))
                R[k]["auc"].append(auroc(zp[m3], zp[m2]))
                row.update(zperp3_mean=round(float(zp[m3].mean()), 4), zperp3_std=round(float(zp[m3].std(ddof=1)), 4),
                           deploy_auroc=round(auroc(zp[m3], zp[m2]), 4))
            ROWS.append(row)
        del bb; torch.cuda.empty_cache()
        print(f"  node_{i} 完成", flush=True)
    mv = lambda k, f: (np.mean(R[k][f]) if R[k][f] else float("nan"))
    print("\n" + "="*104)
    print(f"{'臂':<32}{'到自己類別°':>12}{'②‖z⊥‖':>10}{'②std':>8}{'③‖z⊥‖':>10}{'③std':>8}{'部署AUROC':>11}")
    print("-"*104)
    for k in ARMS:
        print(f"{k:<32}{mv(k,'own'):12.2f}{mv(k,'zp2'):10.4f}{mv(k,'sd2'):8.4f}"
              f"{mv(k,'zp3'):10.4f}{mv(k,'sd3'):8.4f}{mv(k,'auc'):11.4f}")
    print("-"*104)
    lo, base = mv(ARMS[0], "zp2"), mv(ARMS[1], "zp2"); tot = base - lo
    print(f"  可解釋總量（②基準 {base:.4f} − ①下界 {lo:.4f}）= {tot:.4f}")
    for k, nm in [(ARMS[2], "★ 主臂"), (ARMS[3], "控制A"), (ARMS[5], "零點")]:
        print(f"  {nm:<10} ② 降 {base-mv(k,'zp2'):+.4f} ⇒ 解釋 {(base-mv(k,'zp2'))/tot*100:6.1f}%"
              f"   ③ 變化 {mv(k,'zp3')-mv(ARMS[1],'zp3'):+.4f}   AUROC {mv(k,'auc'):.4f}")
    print(f"  控制B（①no-op）：‖z⊥‖ {mv(ARMS[4],'zp2'):.4f} vs 原樣 {lo:.4f}   Δ {mv(ARMS[4],'zp2')-lo:+.4f}")
    if not BN:
        print(f"  ★ 0818 對照（角距離）：①下界 {mv(ARMS[0],'own'):.2f}° (報 29.63) ｜"
              f" ②基準 {mv(ARMS[1],'own'):.2f}° (報 53.57) ｜ 主臂 {mv(ARMS[2],'own'):.2f}° (報 48.48)")
    print("="*104, flush=True)

with open(f"{OUT}/per_node.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(ROWS[0])); w.writeheader(); w.writerows(ROWS)
SUM = []
for basis in ["bn_avg_B", "no_bn_avg"]:
    sub = [r for r in ROWS if r["basis"] == basis]
    arms = sorted({r["arm"] for r in sub}, key=lambda a: a[0])
    g = lambda a, f: [r[f] for r in sub if r["arm"] == a and r[f] != ""]
    lo = np.mean(g(arms[0], "zperp2_mean")); base = np.mean(g(arms[1], "zperp2_mean")); tot = base - lo
    b3 = np.mean(g(arms[1], "zperp3_mean"))
    for a in arms:
        d2 = base - np.mean(g(a, "zperp2_mean"))
        SUM.append(dict(basis=basis, arm=a,
            angle_own_deg=round(float(np.mean(g(a,"angle_own_deg"))),2),
            zperp2_mean=round(float(np.mean(g(a,"zperp2_mean"))),4), zperp2_std=round(float(np.mean(g(a,"zperp2_std"))),4),
            zperp3_mean=(round(float(np.mean(g(a,"zperp3_mean"))),4) if g(a,"zperp3_mean") else ""),
            zperp3_std=(round(float(np.mean(g(a,"zperp3_std"))),4) if g(a,"zperp3_std") else ""),
            deploy_auroc=(round(float(np.mean(g(a,"deploy_auroc"))),4) if g(a,"deploy_auroc") else ""),
            applies_to=("src(①)" if a.startswith(("1.","5.")) else "tgt(②③)"),
            drop2=("n/a" if a.startswith(("1.","5.")) else round(float(d2),4)),
            explained_pct=("n/a" if a.startswith(("1.","5.")) else round(float(d2/tot*100),1)),
            delta3=("n/a" if a.startswith(("1.","5.")) else (round(float(np.mean(g(a,"zperp3_mean"))-b3),4) if g(a,"zperp3_mean") else ""))))
with open(f"{OUT}/summary.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(SUM[0])); w.writeheader(); w.writerows(SUM)
print(f"\nCSV: {OUT}/per_node.csv（{len(ROWS)} 列）與 {OUT}/summary.csv（{len(SUM)} 列）")
