"""R1+R2：造假點之前的兩個閘門（plan: 0825_synthesis_region_feasibility_plan.md）

比喻（與 dany 對齊的講法）：特徵都在單位球面上＝地球表面；每個類別中心是一個極點；
檢測分數＝離最近極點幾度（緯度）。換畫風若只換經度、分數不變；若把樣本往赤道推、就會誤拒。

R1（緯度拆解）：cartoon 比來源域多出來的緯度，是「整團搬家」還是「散開了」？
   搬家 ⇒ 有共同方向可扣除 ⇒ 排除區有希望
   散開 ⇒ 往四面八方 ⇒ 沒有特定方向可扣 ⇒ 排除區無效
   ⚠️ 設計修正（2026-08-25）：plan 原寫「量位移的符號」，但兩個單位向量相差、
      投影到其中一個上恆為負（d·u = μ_t·u − 1 ≤ 0）⇒ 中心層級的符號是幾何必然、無資訊。
      改量「搬家 vs 散開」的分帳，這才有決定力。

R2（子空間維度）：畫風位移佔 128 維裡的幾個方向？
   太少（<10）⇒ 扣掉≈空操作；太多（>60）⇒ 扣掉等於砍半個空間。兩頭都不行。

全程 BN 平均 B（TaskBoard §A 協定）。person 只當量測對象、不進任何合成流程。
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
CKPT_TAG = os.environ.get("CKPT_TAG", "final")
NODES = int(os.environ.get("NODES", "9"))
CK = "exp_result_" + DESC
leave = "cartoon"; UNK = 6; N = 9; NC = 6; DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]; per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in PACS}
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
ang = lambda a, b: np.arccos(np.clip(a @ b if b.ndim == 1 else (a * b).sum(-1), -1 + 1e-7, 1 - 1e-7)) * DEG


@torch.no_grad()
def collect(bb, loader):
    """回傳 (128維投影 L2 已正規化, 標籤)；**保留 person(6)**"""
    Z, Y = [], []
    for b in loader:
        x, y, _ = util.unpack_batch(b)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        _, v = bb.forward_from_layer3(h)
        Z.append(bb.project(v).cpu().numpy()); Y.append(np.asarray(y).flatten())
    return np.concatenate(Z), np.concatenate(Y).astype(int)


def data_centers(Z, Y):
    """由資料算的類別中心 [6,128]（單位向量）"""
    return np.stack([nrm(Z[Y == c].mean(0)) for c in range(NC)])


def common_frac(D):
    """共走比例：位移向量的平均有多少能量 ⇒ 1=全部同方向、0=各走各的"""
    return float((D.mean(0) ** 2).sum() / (D ** 2).sum(1).mean())


AVG = bn_avg(CK, DESC, N=N, ckpt_tag=CKPT_TAG)
print(f"[cfg] {DESC[:60]}… ckpt={CKPT_TAG} nodes={NODES}  BN平均B已套用", flush=True)

acc = {k: [] for k in "src_lat tgt_lat tgt_own_scat move_deg proto_lat_s proto_lat_t proto_lat_p cf_style cf_noise".split()}
STYLE_D = []      # 排除區來源 1：同一模型看三個來源域，同類別中心兩兩相減
for i in range(NODES):
    bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_{CKPT_TAG}.pth"), 6, "cuda")
    apply_bn(bb, AVG)
    Cp = class_centers(bb.prototypes, bb.proto_count).cpu().numpy()      # 訓練學到的原型中心
    Zs, Ys = collect(bb, ld[OWN[i]])                                     # ① 本地來源域
    Zt, Yt = collect(bb, ld[leave])                                      # ②③ cartoon
    SRC = {d: collect(bb, ld[d]) for d in avail}                         # 三個來源域（供 R2）
    del bb

    ms, mt = Ys != UNK, Yt != UNK
    Us = data_centers(Zs[ms], Ys[ms])          # ← 極點（來源域資料中心）
    Ut = data_centers(Zt[mt], Yt[mt])          # cartoon 自己的中心

    # R1：緯度拆解（每類算完再平均）
    acc["src_lat"].append(np.mean([ang(Zs[ms][Ys[ms] == c], Us[c]).mean() for c in range(NC)]))
    acc["tgt_lat"].append(np.mean([ang(Zt[mt][Yt[mt] == c], Us[c]).mean() for c in range(NC)]))
    acc["tgt_own_scat"].append(np.mean([ang(Zt[mt][Yt[mt] == c], Ut[c]).mean() for c in range(NC)]))
    acc["move_deg"].append(np.mean([ang(Ut[c], Us[c]) for c in range(NC)]))

    # §0 自檢：用訓練原型、min over 6 centers（＝detection_score 口徑）
    acc["proto_lat_s"].append(ang(Zs[ms][:, None, :], Cp[None]).min(1).mean())
    acc["proto_lat_t"].append(ang(Zt[mt][:, None, :], Cp[None]).min(1).mean())
    acc["proto_lat_p"].append(ang(Zt[~mt][:, None, :], Cp[None]).min(1).mean())

    # R1b：搬家方向六類是否一致（未正規化的位移向量）
    Dm = np.stack([Ut[c] - Us[c] for c in range(NC)])
    acc["cf_style"].append(common_frac(Dm))
    # 雜訊地板：同一域內偶/奇兩半的類別中心差（沒有真實位移）
    Dn = np.stack([nrm(Zs[ms][Ys[ms] == c][0::2].mean(0)) - nrm(Zs[ms][Ys[ms] == c][1::2].mean(0)) for c in range(NC)])
    acc["cf_noise"].append(common_frac(Dn))

    # R2 素材：同一模型眼中三個來源域的同類別中心兩兩相減
    Cd = {d: data_centers(SRC[d][0][SRC[d][1] != UNK], SRC[d][1][SRC[d][1] != UNK]) for d in avail}
    for a in range(len(avail)):
        for b in range(a + 1, len(avail)):
            for c in range(NC):
                STYLE_D.append(Cd[avail[a]][c] - Cd[avail[b]][c])
    print(f"  node{i}({OWN[i]}) done", flush=True)

M = {k: float(np.mean(v)) for k, v in acc.items()}
print("\n" + "=" * 92)
print("§0 自檢（detection_score 口徑：訓練原型、min over 6 centers）")
print(f"  ①來源域 {M['proto_lat_s']:.1f}°（deploy probe 34.5）  ②cartoon {M['proto_lat_t']:.1f}°（45.6）  ③person {M['proto_lat_p']:.1f}°（66.6）")
print(f"  雜訊地板（同域偶奇兩半的共走比例，無真實位移）＝{M['cf_noise']:.4f}")
print("=" * 92)
print("\n★ R1 緯度拆解（用『資料算的來源域中心』當極點，每類算完再平均）")
print(f"  A ①來源域樣本的緯度（＝來源域散布）        {M['src_lat']:.2f}°")
print(f"  B ②cartoon 樣本的緯度（＝總效果）          {M['tgt_lat']:.2f}°   ⇒ 多出 {M['tgt_lat']-M['src_lat']:+.2f}°")
print(f"  ├ C ②繞【自己】中心的散布                  {M['tgt_own_scat']:.2f}°   ⇒ 散開了 {M['tgt_own_scat']-M['src_lat']:+.2f}°")
print(f"  └ D ②的中心搬離極點                        {M['move_deg']:.2f}°")
print(f"  R1b 搬家方向六類的共走比例                 {M['cf_style']:.4f}（1=同方向、0=各走各的；雜訊地板 {M['cf_noise']:.4f}）")

D = np.stack(STYLE_D)   # ⚠️ 不中心化：共同的畫風方向本身也是要扣的方向
sv = np.linalg.svd(D, compute_uv=False)
en = np.cumsum(sv ** 2) / (sv ** 2).sum()
k50, k90, k95 = [int(np.searchsorted(en, t) + 1) for t in (0.5, 0.9, 0.95)]
print(f"\n★ R2 畫風子空間（{len(D)} 個跨畫風位移向量，128 維空間）")
print(f"  取到 50% 能量需 {k50} 維   90% 需 {k90} 維   95% 需 {k95} 維")
print(f"  前 5 個方向的能量占比：{np.round((sv[:5]**2)/(sv**2).sum(), 4).tolist()}")
print(f"  ⇒ 閘門：<10 維＝扣掉近乎空操作；>60 維＝扣掉等於砍半個空間。實測 90% ＝ {k90} 維")
print("=" * 92)
