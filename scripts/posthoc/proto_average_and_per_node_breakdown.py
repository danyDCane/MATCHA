"""兩個問題一起答（dany 2026-08-19）：

Q1 原型平均有沒有用？在【BN 已平均】的前提下重測——此時原型是節點之間唯一的差異，
   若它有用，這裡是最容易看出來的地方。三種讀出參照：
     P0 各節點自己的 6 個原型（現況）
     P1 9 節點平均後的 6 個原型
     P2 18 格（6 類 × 3 畫風，各畫風的 3 節點先平均），取最近
Q2 誤拒率／部署 AUROC 是不是被 sketch 節點拖累？逐節點群拆開看。
   （BN 平均讓 sketch 在自己域上付出 +9.97° 的代價，遠高於 art +3.37／photo +2.32）

BN 平均一律用【B 合併變異數】版；同時輸出原樣以供對照。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers

PACS = ["art_painting", "cartoon", "photo", "sketch"]
DESC = os.environ.get("RUN_DESC",
    "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix")
CK = "exp_result_" + DESC
leave = "cartoon"; UNK = 6; N = 9; NC = 6; DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]; per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
SH = {"art_painting": "art", "photo": "photo", "sketch": "sketch"}
ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in PACS}

S = [torch.load(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), map_location="cpu",
                weights_only=False)["backbone_state"] for i in range(N)]
BK = [k for k in S[0] if k.endswith("running_mean") or k.endswith("running_var")]
AVG = {}
for k in BK:
    if k.endswith("running_mean"):
        AVG[k] = torch.stack([S[i][k].float() for i in range(N)]).mean(0)
for k in BK:
    if k.endswith("running_var"):
        mk = k.replace("running_var", "running_mean")
        mi = torch.stack([S[i][mk].float() for i in range(N)])
        vi = torch.stack([S[i][k].float() for i in range(N)])
        AVG[k] = (vi + mi ** 2).mean(0) - mi.mean(0) ** 2
# 9 節點的類別中心（用於原型平均）
CEN = np.stack([class_centers(S[i]["prototypes"].float(), S[i]["proto_count"].float()).numpy()
                for i in range(N)])                                  # [9, 6, 128]
del S
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
P_AVG = nrm(CEN.mean(0))                                             # [6,128] 9 節點平均
P_18 = nrm(np.stack([CEN[[i for i in range(N) if OWN[i] == d]].mean(0) for d in avail]))  # [3,6,128]
P_18 = P_18.transpose(1, 0, 2).reshape(NC * len(avail), -1)          # [18,128]
LAB18 = np.tile(np.arange(NC)[:, None], (1, len(avail))).reshape(-1)  # 每格對應的類別


@torch.no_grad()
def feats(bb, loader):
    Z, Y = [], []
    for b in loader:
        x, y, _ = util.unpack_batch(b)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        _, v = bb.forward_from_layer3(h)
        Z.append(bb.project(v).cpu().numpy()); Y.append(np.asarray(y).flatten())
    Z = np.concatenate(Z); return nrm(Z), np.concatenate(Y)


def score(Z, C):
    """reject score = 到最近中心的角度（度）"""
    return np.arccos(np.clip(Z @ C.T, -1 + 1e-7, 1 - 1e-7)).min(1) * DEG


def own_ang(Z, Y, C):
    m = Y != UNK
    a = np.arccos(np.clip(Z[m] @ C.T, -1 + 1e-7, 1 - 1e-7)) * DEG
    return a[np.arange(m.sum()), Y[m].astype(int)].mean()


def axes(s1, s2, s3):
    return (roc_auc_score([0]*len(s1)+[1]*len(s2), np.r_[s1, s2]),
            roc_auc_score([0]*len(s2)+[1]*len(s3), np.r_[s2, s3]),
            float((s2 > np.quantile(s1, 0.95)).mean()))


READOUTS = ["P0 各節點自己的原型", "P1 9節點平均原型", "P2 18格(6類×3畫風)"]
R = {(bn, ro): {"sty": [], "dep": [], "fpr": [], "own1": [], "own2": [], "grp": []}
     for bn in ["原樣", "BN平均"] for ro in READOUTS}

for i in range(N):
    for bn in ["原樣", "BN平均"]:
        bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
        if bn == "BN平均":
            sd = bb.state_dict()
            for k, v in AVG.items():
                sd[k].copy_(v.to(sd[k].device).to(sd[k].dtype))
        Z1, Y1 = feats(bb, ld[OWN[i]]); Z2, Y2 = feats(bb, ld[leave])
        m1, m2 = Y1 != UNK, Y2 != UNK
        for ro in READOUTS:
            if ro.startswith("P0"):
                C = nrm(CEN[i]); s1, s2 = score(Z1, C), score(Z2, C)
                o1, o2 = own_ang(Z1, Y1, C), own_ang(Z2, Y2, C)
            elif ro.startswith("P1"):
                C = P_AVG; s1, s2 = score(Z1, C), score(Z2, C)
                o1, o2 = own_ang(Z1, Y1, C), own_ang(Z2, Y2, C)
            else:
                s1, s2 = score(Z1, P_18), score(Z2, P_18)
                a1 = np.arccos(np.clip(Z1[m1] @ P_18.T, -1+1e-7, 1-1e-7)) * DEG
                a2 = np.arccos(np.clip(Z2[m2] @ P_18.T, -1+1e-7, 1-1e-7)) * DEG
                o1 = np.mean([a1[t, LAB18 == Y1[m1][t]].min() for t in range(len(a1))])
                o2 = np.mean([a2[t, LAB18 == Y2[m2][t]].min() for t in range(len(a2))])
            sty, dep, fpr = axes(s1[m1], s2[m2], s2[~m2])
            d = R[(bn, ro)]
            d["sty"].append(sty); d["dep"].append(dep); d["fpr"].append(fpr)
            d["own1"].append(o1); d["own2"].append(o2); d["grp"].append(SH[OWN[i]])
        del bb
    print(f"  node_{i} ({SH[OWN[i]]}) 完成", flush=True)

m = lambda x: float(np.mean(x))
W = 96
print("\n" + "=" * W); print("§0 自檢"); print("=" * W)
b = R[("原樣", "P0 各節點自己的原型")]
print(f"  原樣 P0：畫風 {m(b['sty']):.4f}  部署 {m(b['dep']):.4f}  誤拒 {m(b['fpr']):.4f}"
      f"   （0818 報 .7938 / .7956 / .4231）")
print(f"  原樣 P0：①{m(b['own1']):.2f}°  ②{m(b['own2']):.2f}°   （既有 29.63 / 53.57）")
b1 = R[("原樣", "P1 9節點平均原型")]
print(f"  原樣 P1（＝0818 §3.3 的 2a）：部署 {m(b1['dep']):.4f} 誤拒 {m(b1['fpr']):.4f}"
      f"   （0818 報 .7952 / .4149）")

print("\n" + "=" * W)
print("★ Q1：原型平均有沒有用")
print("=" * W)
for bn in ["原樣", "BN平均"]:
    print(f"\n  ── {bn} ──")
    print(f"  {'讀出參照':<24}{'①':>9}{'②':>9}{'畫風':>10}{'部署':>10}{'誤拒率':>10}")
    base = None
    for ro in READOUTS:
        d = R[(bn, ro)]; row = (m(d['own1']), m(d['own2']), m(d['sty']), m(d['dep']), m(d['fpr']))
        tail = "" if base is None else f"   Δ部署 {row[3]-base[3]:+.4f}  Δ誤拒 {row[4]-base[4]:+.4f}"
        if base is None: base = row
        print(f"  {ro:<22}{row[0]:>8.2f}°{row[1]:>8.2f}°{row[2]:>10.4f}{row[3]:>10.4f}{row[4]:>10.4f}{tail}")

print("\n" + "=" * W)
print("★ Q2：是不是 sketch 節點在拖後腿（逐節點群，P0 讀出）")
print("=" * W)
for bn in ["原樣", "BN平均"]:
    d = R[(bn, "P0 各節點自己的原型")]
    g = np.array(d["grp"])
    print(f"\n  ── {bn} ──")
    print(f"  {'節點群':<10}{'①':>9}{'②':>9}{'②−①':>9}{'畫風':>10}{'部署':>10}{'誤拒率':>10}")
    for gname in ["art", "photo", "sketch"]:
        k = g == gname
        print(f"  {gname:<10}{np.mean(np.array(d['own1'])[k]):>8.2f}°"
              f"{np.mean(np.array(d['own2'])[k]):>8.2f}°"
              f"{np.mean(np.array(d['own2'])[k])-np.mean(np.array(d['own1'])[k]):>8.2f}°"
              f"{np.mean(np.array(d['sty'])[k]):>10.4f}"
              f"{np.mean(np.array(d['dep'])[k]):>10.4f}"
              f"{np.mean(np.array(d['fpr'])[k]):>10.4f}")
    print(f"  {'全部':<10}{m(d['own1']):>8.2f}°{m(d['own2']):>8.2f}°"
          f"{m(d['own2'])-m(d['own1']):>8.2f}°{m(d['sty']):>10.4f}{m(d['dep']):>10.4f}{m(d['fpr']):>10.4f}")
