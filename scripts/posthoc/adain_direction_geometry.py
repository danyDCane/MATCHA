"""路徑 A 的上界估計：把畫風換掉之後，殘差【方向】的幾何會不會改善？

【問的是什麼】D1 失敗的機制是 ②③ 沿 ū₁ 只差 6.8°（93.7° vs 100.5°）。
若「畫風讓殘差方向轉了近 90°」這個診斷成立，那把 cartoon 的通道統計量換成來源域的
（確定性 AdaIN、作弊上界）應該讓 ② 靠回 ū₁、而 ③ 不跟著靠 ⇒ 角度差拉開、D1 變好。

【判準（事前寫死）】
  主判準  ②③ 沿 ū₁ 的角度差：6.8° → ≥15° 值得往訓練版走｜10–15° 邊際｜<10° 收
  副判準  D1 的部署 AUROC：0.6114 → ≥0.70 支持｜<0.65 收
  健全性  ① 套自己的統計量（no-op）幾何不應變；隨機統計量當零點

⚠️ 射程：0829 已測得 AdaIN 在 BN 平均協定下只觸及 6.8% 的 ‖z⊥‖ 污染
⇒ 本檢驗是**保守下界**。拉不開 ≠ 訓練一定不行；拉得開才是強證據。
基於 adain_intervention_zperp.py（同 checkpoint／同 BN 平均／同 AdaIN 實作）。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "."); sys.path.insert(0, "scripts/posthoc")
import torch, util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
from style_transforms import adain
from bn_common import bn_avg, apply_bn
from scipy.stats import rankdata

PACS = ["art_painting", "cartoon", "photo", "sketch"]
DESC = "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
CK, leave, UNK, N, DEG = "exp_result_" + DESC, "cartoon", 6, 9, 57.29577951308232
avail = [d for d in PACS if d != leave]; per = N // len(avail)
tgt = TD.load_pacs_test_data("../datasets/", leave, 64, 4)[0]
src = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in avail}
def auroc(pos, neg):
    a = np.concatenate([pos, neg]); r = rankdata(a, method="average")
    return float((r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
def ub(u):
    m = u.mean(0); return m/max(np.linalg.norm(m), 1e-8)

@torch.no_grad()
def dom_stats(bb, ld):
    acc = {k: [[], []] for k in ("layer1","layer2","layer3")}
    for b in ld:
        d,_,_ = util.unpack_batch(b); d = d.to("cuda")
        Fm = bb.extract_features_to_layer3(d)
        for k in acc:
            f = Fm[k]; B,C,H,W = f.shape; fl = f.view(B,C,-1)
            acc[k][0].append(fl.mean(2).cpu()); acc[k][1].append(fl.std(2).cpu())
    return {k:(torch.cat(v[0]).mean(0).cuda(), torch.cat(v[1]).mean(0).cuda()) for k,v in acc.items()}

@torch.no_grad()
def collect_u(bb, ld, U, st=None):
    """回傳 (u 方向 [n,128], y)"""
    us, ys = [], []
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
        z = bb.project(vec); z = (z/z.norm(dim=1,keepdim=True)).cpu().numpy().astype(np.float64)
        zp = z - (z @ U) @ U.T
        us.append(zp/np.maximum(np.linalg.norm(zp,axis=1),1e-8)[:,None]); ys.append(y)
    return np.concatenate(us), np.concatenate(ys)

AVG = bn_avg(CK, DESC, N=N)
R = {k: [] for k in ["c12_raw","c12_ada","d2_raw","d3_raw","d2_ada","d3_ada",
                     "gap_raw","gap_ada","auc_raw","auc_ada","c23_raw","c23_ada",
                     "gap_rnd","auc_rnd","c12_noop"]}
for i in range(N):
    bb,_ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
    apply_bn(bb, AVG); bb.eval()
    C = class_centers(bb.prototypes, bb.proto_count)
    Cn = (C/C.norm(dim=1,keepdim=True)).cpu().numpy().astype(np.float64)
    U = np.linalg.svd(Cn.T, full_matrices=False)[0]
    own = avail[min(i//per, len(avail)-1)]
    st_src = dom_stats(bb, src[own])
    g = torch.Generator().manual_seed(2026+i)
    st_rnd = {k:(v[0][torch.randperm(len(v[0]),generator=g).to(v[0].device)],
                 v[1][torch.randperm(len(v[1]),generator=g).to(v[1].device)]) for k,v in st_src.items()}

    u_s, y_s = collect_u(bb, src[own], U);          u1 = ub(u_s[y_s!=UNK])
    u_so, _  = collect_u(bb, src[own], U, st_src)   # 控制B：① 套自己的統計量（no-op）
    R["c12_noop"].append(ub(u_so[y_s!=UNK]) @ u1)
    for tag, st in [("raw", None), ("ada", st_src), ("rnd", st_rnd)]:
        u_t, y_t = collect_u(bb, tgt, U, st)
        m2, m3 = y_t!=UNK, y_t==UNK
        a2 = np.degrees(np.arccos(np.clip(u_t[m2]@u1,-1,1))).mean()
        a3 = np.degrees(np.arccos(np.clip(u_t[m3]@u1,-1,1))).mean()
        s = -(u_t@u1)
        if tag != "rnd":
            R[f"c12_{tag}"].append(ub(u_t[m2])@u1); R[f"c23_{tag}"].append(ub(u_t[m2])@ub(u_t[m3]))
            R[f"d2_{tag}"].append(a2); R[f"d3_{tag}"].append(a3)
        R[f"gap_{tag}"].append(a3-a2); R[f"auc_{tag}"].append(auroc(s[m3], s[m2]))
    del bb; torch.cuda.empty_cache(); print(f"  node_{i} 完成", flush=True)

M = {k: float(np.mean(v)) for k,v in R.items()}
W=92; print("\n"+"="*W); print("AdaIN 換掉畫風之後，殘差【方向】的幾何（BN 平均 B、9 節點）"); print("="*W)
print(f"{'量':<34}{'原樣':>12}{'AdaIN 後':>12}{'Δ':>10}")
print("-"*W)
print(f"{'cos(ū₁, ū₂)  ② 有沒有靠回 ū₁':<34}{M['c12_raw']:+12.4f}{M['c12_ada']:+12.4f}{M['c12_ada']-M['c12_raw']:+10.4f}")
print(f"{'  ↳ 換算夾角':<34}{np.degrees(np.arccos(M['c12_raw'])):11.1f}°{np.degrees(np.arccos(M['c12_ada'])):11.1f}°"
      f"{np.degrees(np.arccos(M['c12_ada']))-np.degrees(np.arccos(M['c12_raw'])):+9.1f}°")
print(f"{'cos(ū₂, ū₃)  兩團之間':<34}{M['c23_raw']:+12.4f}{M['c23_ada']:+12.4f}{M['c23_ada']-M['c23_raw']:+10.4f}")
print("-"*W)
print(f"{'② 到 ū₁ 的平均角度':<34}{M['d2_raw']:11.1f}°{M['d2_ada']:11.1f}°{M['d2_ada']-M['d2_raw']:+9.1f}°")
print(f"{'③ 到 ū₁ 的平均角度':<34}{M['d3_raw']:11.1f}°{M['d3_ada']:11.1f}°{M['d3_ada']-M['d3_raw']:+9.1f}°")
print(f"{'★ 主判準：角度差 (③−②)':<34}{M['gap_raw']:11.1f}°{M['gap_ada']:11.1f}°{M['gap_ada']-M['gap_raw']:+9.1f}°")
print(f"{'★ 副判準：D1 部署 AUROC':<34}{M['auc_raw']:12.4f}{M['auc_ada']:12.4f}{M['auc_ada']-M['auc_raw']:+10.4f}")
print("-"*W)
print(f"{'零點：隨機統計量  角度差':<34}{'—':>12}{M['gap_rnd']:11.1f}°")
print(f"{'零點：隨機統計量  D1 AUROC':<34}{'—':>12}{M['auc_rnd']:12.4f}")
print(f"{'健全性：① 套自己統計量 cos(ū₁,·)':<34}{'—':>12}{M['c12_noop']:+12.4f}  (應≈1)")
print("="*W)
g, a = M['gap_ada'], M['auc_ada']
print(f"\n判定：角度差 {M['gap_raw']:.1f}° → {g:.1f}°  "
      f"{'✅ ≥15° 值得往訓練版走' if g>=15 else ('⚠️ 10–15° 邊際' if g>=10 else '⛔ <10° 收')}")
print(f"      D1 AUROC {M['auc_raw']:.4f} → {a:.4f}  "
      f"{'✅ ≥0.70 支持' if a>=0.70 else ('⚠️ 0.65–0.70 邊際' if a>=0.65 else '⛔ <0.65 收')}")
