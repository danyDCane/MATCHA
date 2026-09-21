"""門檻掃描：在每個門檻之上，實際住著幾張 cartoon 正常圖片、幾張 person？

回答 dany 2026-08-25 的直覺質疑：「②均值45.6、③均值66.6 差這麼多，
把假點放 70° 附近不就好了？」——均值差很多不代表分得開，要看【分布】與【張數】。
全程 BN 平均 B。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
from bn_common import bn_avg, apply_bn

DESC = os.environ.get("DESC", "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix")
CK = "exp_result_" + DESC; N = 9; UNK = 6
ld = TD.load_pacs_test_data("../datasets/", "cartoon", 64, 4)[0]


@torch.no_grad()
def collect(bb, loader):
    Z, Y = [], []
    for b in loader:
        x, y, _ = util.unpack_batch(b)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        _, v = bb.forward_from_layer3(h)
        Z.append(bb.project(v).cpu().numpy()); Y.append(np.asarray(y).flatten())
    return np.concatenate(Z), np.concatenate(Y).astype(int)


AVG = bn_avg(CK, DESC, N=N)
S2, S3 = [], []
for i in range(N):
    bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
    apply_bn(bb, AVG)
    C = class_centers(bb.prototypes, bb.proto_count).cpu().numpy()
    Z, Y = collect(bb, ld); del bb
    s = np.degrees(np.arccos(np.clip(Z @ C.T, -1 + 1e-7, 1 - 1e-7))).min(1)
    S2.append(s[Y != UNK]); S3.append(s[Y == UNK])
    print(f"  node{i} done", flush=True)

n2, n3 = len(S2[0]), len(S3[0])
print(f"\ncartoon 已知類別 {n2} 張   person {n3} 張   （person 佔 {n3/(n2+n3)*100:.1f}%）")
print("=" * 88)
print(f"{'門檻':>6} | {'cartoon 在此之上':>16} | {'person 在此之上':>15} | {'該區 person 佔比':>15} | 說明")
print("-" * 88)
for t in [45, 50, 55, 60, 62.3, 65, 70, 75, 80, 85]:
    c2 = float(np.mean([(s >= t).sum() for s in S2])); c3 = float(np.mean([(s >= t).sum() for s in S3]))
    frac = c3 / max(c2 + c3, 1e-9)
    note = "★ 現行門檻" if abs(t - 62.3) < .01 else ("← 你說的 70°" if t == 70 else "")
    print(f"{t:>6} | {c2:>7.0f} 張 ({c2/n2*100:>4.1f}%) | {c3:>6.0f} 張 ({c3/n3*100:>4.1f}%) | {frac*100:>13.1f}% | {note}")
print("=" * 88)
print(f"無資訊基準（隨便抓一張是 person 的機率）＝ {n3/(n2+n3)*100:.1f}%")
