"""完整 baseline 階梯：從「什麼都沒有」到現在，全部統一在 BN 平均（合併變異數版 B）基底上。

補的缺口：TaskBoard §A 的階梯 A（無 aggbn、有 diffusion、無原型）在 BN 平均協定下**從沒測過**。
四軸定義同 fullspectrum_probe.py：①=該節點來源域已知類別 ②=cartoon 已知類別 ③=cartoon person。
  畫風 AUROC(①vs②) 理想 0.5 ｜ 語意 AUROC(①vs③) ｜ ★部署 AUROC(②vs③) ｜ 誤拒率@src95 ＋ 放行率
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
sys.path.insert(0, "scripts/posthoc")
import torch, util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from bn_common import bn_avg, apply_bn
from dood.prototype import class_centers

PACS = ["art_painting", "cartoon", "photo", "sketch"]
leave, UNK, N, DEG = "cartoon", 6, 9, 57.29577951308232
avail = [d for d in PACS if d != leave]; OWN = [avail[i // (N // len(avail))] for i in range(N)]
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)

LADDER = [
    ("A 最原始（有 diffusion、無 BN 聚合、無原型）",
     "v1_stage2_leave_cartoon_async_const_tau1e-5_style_osdg_excl_person_seed2026_topo1234"),
    ("A+ 只加訓練時 BN 聚合（單一變因）",
     "v1_stage2_leave_cartoon_async_const_tau1e-5_style_aggbn_osdg_excl_person_seed2026_topo1234"),
    ("B 再拿掉 diffusion＝λ=0（論文基準）",
     "v1_stage2_leave_cartoon_proto_lam0_m0.95w10_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"),
    ("C/D 我們的模型（1a-fix）",
     "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"),
]
LD = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in PACS}

def auroc(pos, neg):
    a = np.concatenate([pos, neg]); r = a.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

@torch.no_grad()
def collect(bb, loader):
    Z, Y, E, M = [], [], [], []
    for b in loader:
        x, y, _ = util.unpack_batch(b)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        lo, v = bb.forward_from_layer3(h)
        E.append((-torch.logsumexp(lo, 1)).cpu().numpy())
        M.append((-lo.softmax(1).max(1).values).cpu().numpy())
        try:    Z.append(bb.project(v).cpu().numpy())
        except Exception: Z.append(np.zeros((len(y), 1), np.float32))
        Y.append(np.asarray(y).flatten())
    return np.concatenate(Z), np.concatenate(Y), np.concatenate(E), np.concatenate(M)

print("="*104)
print("完整 baseline 階梯（cartoon fold、seed2026、final、9 節點 node-mean、★全部先做 BN 平均 B）")
print("="*104)
hdr = f"{'階梯':<40} {'讀出':<8} {'畫風↓0.5':>9} {'語意↑':>7} {'★部署↑':>8} {'誤拒率↓':>8} {'放行率↓':>8}"
for tag, DESC in LADDER:
    CK = "exp_result_" + DESC
    AVG = bn_avg(CK, DESC, N=N)
    acc = {}
    for i in range(N):
        bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
        apply_bn(bb, AVG); bb.eval()
        Zs, Ys, Es, Ms = collect(bb, LD[OWN[i]])
        Zt, Yt, Et, Mt = collect(bb, LD[leave])
        ms = Ys != UNK; k2, k3 = Yt != UNK, Yt == UNK
        readouts = {"energy": (Es[ms], Et[k2], Et[k3]), "msp": (Ms[ms], Mt[k2], Mt[k3])}
        if Zt.shape[1] > 1:
            try:
                C = nrm(class_centers(bb.prototypes, bb.proto_count).cpu().numpy().astype(np.float64))
                f = lambda Z: np.arccos(np.clip(nrm(Z.astype(np.float64)) @ C.T, -1+1e-12, 1-1e-12)).min(1)*DEG
                readouts["proto"] = (f(Zs)[ms], f(Zt)[k2], f(Zt)[k3])
            except Exception as e: print("  (proto 讀出不可用:", type(e).__name__, ")")
        for nm_, (a, b, c) in readouts.items():
            tau = np.quantile(a, 0.95)
            acc.setdefault(nm_, []).append([auroc(b, a), auroc(c, a), auroc(c, b),
                                            float((b > tau).mean()), float((c <= tau).mean())])
        del bb; torch.cuda.empty_cache()
    if tag == LADDER[0][0]: print(hdr); print("-"*104)
    for nm_ in ["energy", "msp", "proto"]:
        if nm_ not in acc: continue
        m = np.mean(acc[nm_], 0)
        print(f"{tag if nm_=='energy' else '':<40} {nm_:<8} {m[0]:9.4f} {m[1]:7.4f} {m[2]:8.4f} {m[3]:8.4f} {m[4]:8.4f}")
    print("-"*104)
print("★ 讀法：畫風 AUROC 越接近 0.5 越好（畫風不該影響分數）；部署 AUROC＝主指標；誤拒率與放行率必須成對讀。")
