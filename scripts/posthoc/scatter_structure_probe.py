"""散開那一塊有沒有結構？——cartoon 比來源域多散的 9.6°，是「同樣形狀變大」還是「多了特定方向」？

背景（2026-08-25）：cartoon 跑出城市外＝搬家 31.3°（位置、已排除）＋散開 40.8°（壓緊）。
搬家死了；散開沒被位置路線擋住。但散開可不可治，取決於它有沒有方向結構：
  來源域也是 67 個方向 ⇒ 純尺度問題 ⇒ 回到「力道」，已證無解
  來源域少很多       ⇒ cartoon 多出的是特定新方向 ⇒ 那些方向就是可治的目標

量法：每類先減掉自己的中心（去掉搬家），再對殘差做 SVD，比較兩邊的方向數與主方向重合度。
全程 BN 平均 B。person 只當對照、不參與。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from bn_common import bn_avg, apply_bn

PACS = ["art_painting", "cartoon", "photo", "sketch"]
DESC = os.environ.get("DESC", "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix")
CK = "exp_result_" + DESC; N = 9; NC = 6; UNK = 6; DEG = 57.29577951308232
leave = "cartoon"; avail = [d for d in PACS if d != leave]; per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in PACS}
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)


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


def resid(Z, Y):
    """每類減掉自己的中心方向（去掉搬家），回傳殘差矩陣"""
    out = []
    for c in range(NC):
        X = Z[Y == c]
        if len(X) < 2: continue
        u = nrm(X.mean(0))
        out.append(X - np.outer(X @ u, u))
    return np.concatenate(out)


def dims(R):
    sv = np.linalg.svd(R, compute_uv=False); en = np.cumsum(sv ** 2) / (sv ** 2).sum()
    return [int(np.searchsorted(en, t) + 1) for t in (0.5, 0.9, 0.95)], sv


def scat(Z, Y):
    return float(np.mean([np.degrees(np.arccos(np.clip(Z[Y == c] @ nrm(Z[Y == c].mean(0)), -1 + 1e-7, 1 - 1e-7))).mean()
                          for c in range(NC)]))


AVG = bn_avg(CK, DESC, N=N)
A = {k: [] for k in "d_s d_t d_p sc_s sc_t ov90 ex_frac ex_dim".split()}
for i in range(N):
    bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
    apply_bn(bb, AVG)
    Zs, Ys = collect(bb, ld[OWN[i]]); Zt, Yt = collect(bb, ld[leave]); del bb
    ms, mt = Ys != UNK, Yt != UNK
    Rs, Rt = resid(Zs[ms], Ys[ms]), resid(Zt[mt], Yt[mt])
    (ds, svs), (dt, svt) = dims(Rs), dims(Rt)
    A["d_s"].append(ds); A["d_t"].append(dt)
    A["sc_s"].append(scat(Zs[ms], Ys[ms])); A["sc_t"].append(scat(Zt[mt], Yt[mt]))

    # 主方向重合度：兩邊各取 90% 能量的子空間，量它們的主夾角餘弦平均
    Es = np.linalg.svd(Rs, full_matrices=False)[2][:ds[1]].T
    Et = np.linalg.svd(Rt, full_matrices=False)[2][:dt[1]].T
    A["ov90"].append(float((np.linalg.svd(Es.T @ Et, compute_uv=False) ** 2).sum() / min(ds[1], dt[1])))

    # cartoon 殘差裡「來源域子空間裝不下」的能量占比 ＋ 它自己要幾個方向
    Rex = Rt - (Rt @ Es) @ Es.T
    A["ex_frac"].append(float((Rex ** 2).sum() / (Rt ** 2).sum()))
    A["ex_dim"].append(dims(Rex)[0][1])
    print(f"  node{i}({OWN[i]}) done", flush=True)

m = lambda k: np.mean(A[k], axis=0)
print("\n" + "=" * 90)
print("★ 散開的方向數（每類已減掉自己的中心＝去掉搬家）")
print(f"  ①來源域   50%需 {m('d_s')[0]:.0f} 維   90%需 {m('d_s')[1]:.0f} 維   95%需 {m('d_s')[2]:.0f} 維   （散布 {m('sc_s'):.2f}°）")
print(f"  ②cartoon  50%需 {m('d_t')[0]:.0f} 維   90%需 {m('d_t')[1]:.0f} 維   95%需 {m('d_t')[2]:.0f} 維   （散布 {m('sc_t'):.2f}°）")
print(f"\n★ 兩邊主方向的重合度（1＝完全同一組方向、0＝完全不同）  {m('ov90'):.4f}")
print(f"★ cartoon 散開中【來源域子空間裝不下】的能量占比            {m('ex_frac')*100:.1f}%")
print(f"   那些裝不下的部分自己要幾個方向（90%）                    {m('ex_dim'):.0f} 維")
print("=" * 90)
print("判讀：重合度高 + 裝不下的占比低 ⇒ 同樣形狀變大＝純尺度＝回到力道（已證無解）")
print("      裝不下的占比高 + 它自己維度低 ⇒ 有特定新方向＝可治的目標")
