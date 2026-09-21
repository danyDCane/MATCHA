"""實驗②：把原型自己的 6 個距離榨乾——`min_c` 有沒有漏掉東西？

與實驗① 的差別：① 問「原型分數 + energy 混起來會不會更好」（答案：不會，最佳權重 w=0）；
② 問「原型自己的距離輪廓裡，`min_c` 有沒有漏掉資訊」——**全程不碰 energy**。

直覺：已知類別樣本貼著自己那一類、遠離其餘五類 ⇒ 輪廓陡、變異大；
      person 跟六類都不太像 ⇒ 輪廓平、變異小。而 `energy=logsumexp` 本來就在讀整條輪廓。

⚠️ **公平性控制**：若允許原型用輪廓統計量升級，**energy 也必須拿到同等待遇**
（同表跑 MaxLogit／MSP／logit 輪廓變異數版），否則就是先前批評過的不公平比較。

判準（事前寫死）：
  (b) 成立 ＝ 某變體 部署 AUROC > 0.8380（且贏 energy 的最佳輪廓變體）
            **且** 在相同放行率下誤拒率贏 0.1901（1a-fix energy）與 0.1896（λ=0 energy）
  (b) 收掉 ＝ 最佳變體仍 < 0.8380 ⇒ `min_c` 沒漏掉東西，讀出形式不是問題 ⇒ 全力轉 (a) 幾何

自檢：α=0 / V0 必須逐位重現 0.8145。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers

PACS = ["art_painting", "cartoon", "photo", "sketch"]
RUNS = {"1a-fix": "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix",
        "λ=0": "v1_stage2_leave_cartoon_proto_lam0_m0.95w10_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"}
leave = "cartoon"; UNK = 6; N = 9; DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]; per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in PACS}
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
TARGET_MISS = 0.3572


def bn_avg(CK, DESC):
    S = [torch.load(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), map_location="cpu",
                    weights_only=False)["backbone_state"] for i in range(N)]
    BKs = [k for k in S[0] if k.endswith("running_mean") or k.endswith("running_var")]
    A = {}
    for k in BKs:
        if k.endswith("running_mean"):
            A[k] = torch.stack([S[i][k].float() for i in range(N)]).mean(0)
    for k in BKs:
        if k.endswith("running_var"):
            mk = k.replace("running_var", "running_mean")
            mi = torch.stack([S[i][mk].float() for i in range(N)])
            vi = torch.stack([S[i][k].float() for i in range(N)])
            A[k] = (vi + mi ** 2).mean(0) - mi.mean(0) ** 2
    del S
    return A


@torch.no_grad()
def collect(bb, loader, C):
    """回傳 (角距離矩陣[N,6], logit矩陣[N,6], 標籤)"""
    A, L, Y = [], [], []
    for b in loader:
        x, y, _ = util.unpack_batch(b)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        lo, v = bb.forward_from_layer3(h)
        z = nrm(bb.project(v).cpu().numpy())
        A.append(np.arccos(np.clip(z @ C.T, -1 + 1e-7, 1 - 1e-7)) * DEG)
        L.append(lo.cpu().numpy()); Y.append(np.asarray(y).flatten())
    return np.concatenate(A), np.concatenate(L), np.concatenate(Y)


def ent(M, T):
    p = torch.softmax(torch.from_numpy(M / T), dim=1).numpy()
    return -(p * np.log(p + 1e-12)).sum(1)


def proto_variants(D):
    """D:[n,6] 角距離。回傳 {名稱: 分數}，分數高＝越像 OOD"""
    mn = D.min(1); sd = D.std(1); mean = D.mean(1)
    srt = np.sort(D, 1); marg = srt[:, 1] - srt[:, 0]
    out = {"V0 min（現況）": mn}
    for a in [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]:
        out[f"V1 min − {a}·std"] = mn - a * sd
    for a in [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]:
        out[f"V2 min − {a}·margin"] = mn - a * marg
    out["V3 min − mean（0815 排除過）"] = mn - mean
    out["V4 min / mean"] = mn / np.maximum(mean, 1e-9)
    for T in [2.0, 5.0, 10.0, 20.0]:
        out[f"V5 熵(T={T})"] = -ent(-D, T)      # 輪廓越平 ⇒ 熵越大 ⇒ 越像 OOD ⇒ 取負再定向
    return out


def logit_variants(L):
    """公平性控制：energy 側也給同等待遇。分數高＝越像 OOD"""
    t = torch.from_numpy(L)
    e = (-torch.logsumexp(t, 1)).numpy()
    mx = (-t.max(1).values).numpy()
    ms = (-t.softmax(1).max(1).values).numpy()
    sd = L.std(1)
    out = {"E0 energy": e, "E1 MaxLogit": mx, "E2 MSP": ms}
    for a in [0.1, 0.3, 0.5, 1.0]:
        out[f"E3 energy − {a}·std(logit)"] = e - a * sd
    return out


ACC = {}
for tag, DESC in RUNS.items():
    CK = "exp_result_" + DESC
    AVG = bn_avg(CK, DESC)
    rec = []
    for i in range(N):
        bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
        sd_ = bb.state_dict()
        for k, v in AVG.items(): sd_[k].copy_(v.to(sd_[k].device).to(sd_[k].dtype))
        C = nrm(class_centers(bb.prototypes, bb.proto_count).cpu().numpy())
        D1, L1, y1 = collect(bb, ld[OWN[i]], C)
        D2, L2, y2 = collect(bb, ld[leave], C)
        del bb
        k1, k2, u2 = y1 != UNK, y2 != UNK, y2 == UNK
        rec.append(dict(D=(D1[k1], D2[k2], D2[u2]), L=(L1[k1], L2[k2], L2[u2])))
        print(f"  [{tag}] node_{i} 完成", flush=True)
    ACC[tag] = rec


def evaluate(rec, kind):
    """回傳 {變體名: (部署AUROC, 固定放行率下的誤拒率)}"""
    names = list((proto_variants if kind == "D" else logit_variants)(
        rec[0][kind][0] if kind == "D" else rec[0][kind][0]).keys())
    res = {n: [[], []] for n in names}
    for d in rec:
        f = proto_variants if kind == "D" else logit_variants
        s1, s2, s3 = f(d[kind][0]), f(d[kind][1]), f(d[kind][2])
        for n in names:
            res[n][0].append(roc_auc_score([0]*len(s2[n])+[1]*len(s3[n]), np.r_[s2[n], s3[n]]))
            tau = float(np.quantile(s3[n], TARGET_MISS))
            res[n][1].append(float((s2[n] > tau).mean()))
    return {n: (float(np.mean(v[0])), float(np.mean(v[1]))) for n, v in res.items()}


W = 92
R1 = evaluate(ACC["1a-fix"], "D"); E1 = evaluate(ACC["1a-fix"], "L")
R0 = evaluate(ACC["λ=0"], "D");   E0 = evaluate(ACC["λ=0"], "L")

print("\n" + "=" * W); print("§0 自檢"); print("=" * W)
print(f"  1a-fix V0 min（現況）      部署 {R1['V0 min（現況）'][0]:.4f}（應為 0.8145）"
      f"  誤拒@放行{TARGET_MISS} {R1['V0 min（現況）'][1]:.4f}（應為 0.2212）")
print(f"  1a-fix E0 energy           部署 {E1['E0 energy'][0]:.4f}（應為 0.8380）"
      f"  誤拒@同操作點 {E1['E0 energy'][1]:.4f}（應為 0.1901）")
print(f"  λ=0    E0 energy           部署 {E0['E0 energy'][0]:.4f}（應為 0.8337）"
      f"  誤拒@同操作點 {E0['E0 energy'][1]:.4f}（應為 0.1896）")

print("\n" + "=" * W)
print("★ 原型側：從 6 個距離裡能不能榨出更多（1a-fix）")
print("=" * W)
print(f"  {'變體':<30}{'部署 AUROC':>13}{'vs 0.8380':>12}{'誤拒@同操作點':>16}{'vs 0.1901':>12}")
for n, (a, f) in sorted(R1.items(), key=lambda kv: -kv[1][0]):
    mark = " ★" if (a > 0.8380 and f < 0.1901) else ""
    print(f"  {n:<30}{a:>13.4f}{a-0.8380:>+12.4f}{f:>16.4f}{f-0.1901:>+12.4f}{mark}")

print("\n" + "=" * W)
print("★ 公平性控制：energy 側拿到同等待遇（1a-fix）")
print("=" * W)
print(f"  {'變體':<30}{'部署 AUROC':>13}{'誤拒@同操作點':>16}")
for n, (a, f) in sorted(E1.items(), key=lambda kv: -kv[1][0]):
    print(f"  {n:<30}{a:>13.4f}{f:>16.4f}")

print("\n" + "=" * W)
print("★ 對 λ=0 做同樣處理（規則：每個改動都要對 baseline 做一次再比）")
print("=" * W)
b1 = max(R1.items(), key=lambda kv: kv[1][0]); b0 = max(R0.items(), key=lambda kv: kv[1][0])
be1 = max(E1.items(), key=lambda kv: kv[1][0]); be0 = max(E0.items(), key=lambda kv: kv[1][0])
print(f"  1a-fix 原型最佳：{b1[0]:<26}部署 {b1[1][0]:.4f}  誤拒 {b1[1][1]:.4f}")
print(f"  λ=0    原型最佳：{b0[0]:<26}部署 {b0[1][0]:.4f}  誤拒 {b0[1][1]:.4f}")
print(f"  1a-fix logit最佳：{be1[0]:<25}部署 {be1[1][0]:.4f}  誤拒 {be1[1][1]:.4f}")
print(f"  λ=0    logit最佳：{be0[0]:<25}部署 {be0[1][0]:.4f}  誤拒 {be0[1][1]:.4f}")

print("\n" + "=" * W); print("★ 判決"); print("=" * W)
win = b1[1][0] > 0.8380 and b1[1][0] > be1[1][0] and b1[1][1] < 0.1901 and b1[1][1] < 0.1896
print(f"  原型最佳變體 部署 {b1[1][0]:.4f} vs 目標 0.8380 ⇒ {'✅ 過' if b1[1][0] > 0.8380 else '❌ 未過'}")
print(f"  原型最佳變體 vs energy 同等待遇後的最佳 {be1[1][0]:.4f} ⇒ {'✅ 贏' if b1[1][0] > be1[1][0] else '❌ 輸'}")
print(f"  誤拒@同操作點 {b1[1][1]:.4f} vs 0.1901/0.1896 ⇒ {'✅ 贏' if win else '❌ 未全贏'}")
print(f"\n  ⇒ 路線(b) {'【成立】讀出形式就是那 0.0235 的來源' if win else '【收掉】min_c 沒漏掉東西，讀出形式不是問題 ⇒ 全力轉 (a) 幾何'}")
