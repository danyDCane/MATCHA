"""R2b + R3pre + R3：合成區裡實際住著誰（plan 0825 第二段）

R2b   cartoon 繞自己中心的【散布】集中在幾個方向？（R2 量的是中心搬家，散布才是體積來源）
R3pre 不用造點就能問：把門檻設在 θ_lo，被判異常的樣本裡 ② 與 ③ 各佔多少？
      ⇒ 這就是「合成區的居民組成」，是造點會撞到誰的下界。
R3    造點四臂（臂0 不扣方向／臂1 扣跨節點畫風方向），量每個假點最近的真實鄰居是 ② 還是 ③。

全程 BN 平均 B。person 只當量測對象，不進造點流程。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
from bn_common import bn_avg, apply_bn

PACS = ["art_painting", "cartoon", "photo", "sketch"]
DESC = os.environ.get("DESC", "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix")
CKPT_TAG = os.environ.get("CKPT_TAG", "final"); NODES = int(os.environ.get("NODES", "9"))
THETA_LO = float(os.environ.get("THETA_LO", "62.3")); M_PER_CLS = int(os.environ.get("M", "2000"))
KDIM = int(os.environ.get("KDIM", "7"))
CK = "exp_result_" + DESC
leave = "cartoon"; UNK = 6; N = 9; NC = 6; DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]; per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in PACS}
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
rng = np.random.default_rng(2026)


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


def dcen(Z, Y):
    return np.stack([nrm(Z[Y == c].mean(0)) for c in range(NC)])


def score(Z, C):
    """detection_score 口徑：到最近類別中心的角度（度）"""
    return np.degrees(np.arccos(np.clip(Z @ C.T, -1 + 1e-7, 1 - 1e-7))).min(1)


def synth(C, E, m):
    """在每個類別中心周圍、緯度 [θ_lo, 90°] 的球帶上造點；E 非空則先扣掉該子空間"""
    out = []
    for c in range(NC):
        u = C[c]
        v = rng.standard_normal((m, 128))
        v -= np.outer(v @ u, u)
        if E is not None and E.shape[1] > 0:
            v -= (v @ E) @ E.T
        v = nrm(v)
        th = np.radians(rng.uniform(THETA_LO, 90.0, size=(m, 1)))
        z = np.cos(th) * u + np.sin(th) * v
        z = nrm(z)
        out.append(z[score(z, C) >= THETA_LO - 1e-6])       # 必須離「每一個」中心都夠遠
    return np.concatenate(out)


AVG = bn_avg(CK, DESC, N=N, ckpt_tag=CKPT_TAG)
print(f"[cfg] θ_lo={THETA_LO}° M={M_PER_CLS}/類 KDIM={KDIM} nodes={NODES}  BN平均B", flush=True)
A = {k: [] for k in "sc_dim2 sc_dim5 sc_dim9 p2_over p3_over reg_p3 null_p3 nn0_p3 nn1_p3 keep0 keep1 proj0 proj1".split()}

for i in range(NODES):
    bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_{CKPT_TAG}.pth"), 6, "cuda")
    apply_bn(bb, AVG)
    Cp = class_centers(bb.prototypes, bb.proto_count).cpu().numpy()
    Zs, Ys = collect(bb, ld[OWN[i]])
    Zt, Yt = collect(bb, ld[leave])
    SRC = {d: collect(bb, ld[d]) for d in avail}
    del bb
    m2, m3 = Yt != UNK, Yt == UNK
    Z2, Y2, Z3 = Zt[m2], Yt[m2], Zt[m3]

    # R2b：cartoon 散布的方向維度（每類先減自己中心）
    Ut = dcen(Z2, Y2)
    Rk = np.concatenate([Z2[Y2 == c] - np.outer(Z2[Y2 == c] @ Ut[c], Ut[c]) for c in range(NC)])
    sv = np.linalg.svd(Rk, compute_uv=False); en = np.cumsum(sv ** 2) / (sv ** 2).sum()
    for t, k in [(0.5, "sc_dim2"), (0.9, "sc_dim5"), (0.95, "sc_dim9")]:
        A[k].append(int(np.searchsorted(en, t) + 1))

    # R3pre：門檻 θ_lo 之上的居民組成
    s2, s3 = score(Z2, Cp), score(Z3, Cp)
    o2, o3 = (s2 >= THETA_LO).sum(), (s3 >= THETA_LO).sum()
    A["p2_over"].append(o2 / len(s2)); A["p3_over"].append(o3 / len(s3))
    A["reg_p3"].append(o3 / max(o2 + o3, 1))
    A["null_p3"].append(len(s3) / (len(s2) + len(s3)))

    # 畫風子空間 E（跨節點同類別中心兩兩相減）
    Cd = {d: dcen(SRC[d][0][SRC[d][1] != UNK], SRC[d][1][SRC[d][1] != UNK]) for d in avail}
    D = np.stack([Cd[avail[a]][c] - Cd[avail[b]][c]
                  for a in range(len(avail)) for b in range(a + 1, len(avail)) for c in range(NC)])
    E = np.linalg.svd(D, full_matrices=False)[2][:KDIM].T          # [128, KDIM] 正交基

    # R3：造點四臂（本段：臂0 不扣、臂1 扣畫風方向）→ 最近鄰身分
    REAL = np.concatenate([Z2, Z3]); LAB = np.concatenate([np.zeros(len(Z2)), np.ones(len(Z3))])
    for tag, Ei in [("0", None), ("1", E)]:
        S = synth(Cp, Ei, M_PER_CLS)
        A["keep" + tag].append(len(S) / (NC * M_PER_CLS))
        A["proj" + tag].append(float((((S @ E) ** 2).sum(1)).mean()))   # 落在畫風子空間的能量
        nn = LAB[np.argmax(S @ REAL.T, axis=1)]
        A["nn" + tag + "_p3"].append(float(nn.mean()))
    print(f"  node{i}({OWN[i]}) done", flush=True)

M_ = {k: float(np.mean(v)) for k, v in A.items()}
print("\n" + "=" * 94)
print(f"★ R2b cartoon【散布】的方向維度（128 維空間）：50%需 {M_['sc_dim2']:.0f} 維  90%需 {M_['sc_dim5']:.0f} 維  95%需 {M_['sc_dim9']:.0f} 維")
print(f"   （對照 R2 中心搬家：50%需2維、90%需7維）")
print(f"\n★ R3pre 門檻 {THETA_LO}° 之上的居民組成（不需造點）")
print(f"   ②cartoon 已知類別 超過門檻的比例  {M_['p2_over']*100:.1f}%")
print(f"   ③person           超過門檻的比例  {M_['p3_over']*100:.1f}%")
print(f"   ⇒ 門檻之上 person 佔 {M_['reg_p3']*100:.1f}%   （無資訊基準＝總體 person 占比 {M_['null_p3']*100:.1f}%）")
print(f"\n★ R3 造點後最近鄰是 person 的比例（越高越好；基準 {M_['null_p3']*100:.1f}%）")
print(f"   臂0 不扣畫風方向   p3 = {M_['nn0_p3']*100:.1f}%   合格率 {M_['keep0']*100:.1f}%   落在畫風子空間能量 {M_['proj0']:.4f}")
print(f"   臂1 扣掉{KDIM}維畫風方向 p3 = {M_['nn1_p3']*100:.1f}%   合格率 {M_['keep1']*100:.1f}%   落在畫風子空間能量 {M_['proj1']:.4f}")
print("=" * 94)
