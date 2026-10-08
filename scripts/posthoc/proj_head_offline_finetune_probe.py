"""離線預測「原型跨節點聚合會不會傷檢測成績」——只微調投影層，骨幹凍結。

★ 為什麼可以這樣做：2b 改的東西只從 `comp_loss`／`disp_loss` 進來，而它們只看 `proj_head`
  輸出的 128 維向量。凍結骨幹、只訓練那兩層線性，就能用極低成本模擬「換了目標之後特徵被推去哪」。

★ 四個臂（**唯一差別是損失定義**，初始值／抽樣順序／種子全同 ⇒ 配對比較）
  arm1 現況      6 格原型（只有自己畫風那欄）、comp + disp
  arm2 完整 2b   18 格、每樣本 3 個正樣本
  arm3 設計 A    18 格、遮罩三值：自己畫風同類別＝正｜其他類別（三畫風）＝負｜**同類別其他畫風＝忽略**
  arm4 comp-only disp 關掉 ⇒ **方向錨點**（真實從頭訓練的已知答案：cartoon disp 值 +0.0318±0.0063、9/9 同號）

★★ 三層驗證（**全部不過就不該相信任何臂的數字**）
  驗證1 起點錨：step 0 的九節點平均部署 AUROC 必須重現登記值（點 0.8145／面 0.8281）
  驗證2 解析度：arm1（目標沒變）的漂移若大於我們要偵測的效應 ⇒ 探針沒有解析度
  驗證3 方向錨：arm4 必須比 arm1 差（真實從頭訓練是 −0.0318）

⛔ **原本承諾的 `L_pair` 錨點跑不了**：`pair_loss = relu(margin − cos(z_擾動, z_乾淨))`，
   離線只有一組特徵 ⇒ cos ≡ 1 ⇒ 損失恆為 0、無梯度。風格增強發生在 layer3 的空間特徵圖上，
   從池化後的 512 維向量無法重建。

⚠️ 與真實訓練的四個偏離（讀結果時必須一起讀）
  1. 骨幹凍結（真實訓練骨幹也在動、梯度會往前傳）
  2. `comp_loss` 真實吃的是**風格擾動後**的 z_aug，本檔只有乾淨 z ⇒ 用乾淨的代替
  3. 原型 EMA 用**閉式向量化版**（跳過逐樣本的中間正規化），非逐樣本序列 ⇒ 數值會與真實微漂
  4. 輸入特徵來自**終態** checkpoint ⇒ 這是「從現在的解出發的局部預測」，不是從頭訓練
  ⇒ **結論只能用來排順序、不能用來判死。**

⚠️ 評估端刻意固定：四個臂**一律用自己畫風那欄算出的六個中心**當讀出參照物
   （matcha-02 四折已測「事後換參照物」≈0）⇒ 變的只有「訓練把特徵推到哪」。

⚠️ 獨立性：`0826_features_full.npz` 餵給每個節點的是**整個域**（matcha-02 2026-10-01 查證）
   ⇒ 同畫風三節點特徵完全相同 ⇒ **訓練只跑 3 個代表節點（0/3/6），step-0 錨點跑全 9 個。**
"""
import os, sys, numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dood.prototype import class_centers, residual_projector, residual_score, detection_score

R = ("v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_"
     "osdg_excl_person_seed2026_topo1234_fix")
D, N, NC, UNK = f"exp_result_{R}", 9, 6, 6
LR, MOM, BS, STEPS, EVERY = 1e-3, 0.9, 64, 4000, 500      # 對齊 args：lr=0.001 momentum=0.9
PROTO_M, T, RC, RD = 0.95, 0.1, 0.1, 0.1                  # 對齊 args
dev = "cuda" if torch.cuda.is_available() else "cpu"
F = np.load("logs/prototype_probe/0826_features_full.npz")
nrm = lambda X: Fn.normalize(X, dim=-1)

def auroc(pos, neg):
    a = torch.cat([pos, neg]); r = a.argsort().argsort().double() + 1
    n, m = len(pos), len(neg)
    return float((r[:n].sum() - n * (n + 1) / 2) / (n * m))

def load(i):
    sd = torch.load(f"{D}/{R}_node_{i}_final.pth", map_location="cpu",
                    weights_only=False)["backbone_state"]
    W = [sd["proj_head.0.weight"].clone(), sd["proj_head.0.bias"].clone(),
         sd["proj_head.2.weight"].clone(), sd["proj_head.2.bias"].clone()]
    return [w.double().to(dev) for w in W], sd["prototypes"].double().to(dev), \
           sd["proto_count"].double().to(dev)

def proj(W, h):          # ＝ models/resnet.py 的 project()
    return nrm(Fn.linear(torch.relu(Fn.linear(h, W[0], W[1])), W[2], W[3]))

SRC_H = {i: torch.tensor(F[f"n{i}_src_h"].astype(np.float64), device=dev) for i in range(N)}
SRC_Y = {i: torch.tensor(F[f"n{i}_src_y"].astype(np.int64), device=dev) for i in range(N)}
TGT_H = {i: torch.tensor(F[f"n{i}_tgt_h"].astype(np.float64), device=dev) for i in range(N)}
TGT_Y = {i: torch.tensor(F[f"n{i}_tgt_y"].astype(np.int64), device=dev) for i in range(N)}

# ── 驗證1：起點錨點（全 9 節點，不訓練）──────────────────────────────────────
pt0, pl0 = [], []
BASE = {}
for i in range(N):
    W, pr, cn = load(i); BASE[i] = (W, pr, cn)
    z = proj(W, TGT_H[i]); y = TGT_Y[i]
    C = class_centers(pr, cn)
    m3, m2 = y == UNK, y != UNK
    pt0.append(auroc(detection_score(z, C)[m3], detection_score(z, C)[m2]))
    P = residual_projector(C)
    pl0.append(auroc(residual_score(z, P)[m3], residual_score(z, P)[m2]))
print("=" * 96)
print("驗證1 起點錨點（9 節點平均、未訓練）")
print("=" * 96)
print(f"  點讀出 proto_angle  {np.mean(pt0):.4f}   登記值 0.8145   差 {np.mean(pt0)-0.8145:+.4f}")
print(f"  面讀出 ‖z⊥‖        {np.mean(pl0):.4f}   登記值 0.8281   差 {np.mean(pl0)-0.8281:+.4f}")
ok1 = abs(np.mean(pt0) - 0.8145) < 5e-4 and abs(np.mean(pl0) - 0.8281) < 5e-4
print(f"  ⇒ {'✅ 通過' if ok1 else '🚨 沒通過 ⇒ 管線有問題，後面的數字不要信'}")
if not ok1:
    sys.exit(1)

# 其他兩個畫風的原型（固定不動），供 arm2／arm3 使用
OWNP = {i: BASE[i][1][torch.arange(NC), (BASE[i][2] > 0).double().argmax(1)] for i in range(N)}
P3 = torch.stack([nrm(torch.stack([OWNP[k] for k in range(N) if k // 3 == d]).mean(0))
                  for d in range(3)])                                    # [3,6,128]

def comp(z, y, flat, pcls, psty, own, mode):
    lg = (z @ flat.t()) / T
    lg = lg - lg.max(1, keepdim=True).values.detach()
    same = y.view(-1, 1) == pcls.view(1, -1)
    pos = same & (psty.view(1, -1) == own) if mode == "A" else same
    neg = (~same).double()
    ex = lg.exp(); ns = (ex * neg).sum(1, keepdim=True)
    lp = lg - torch.log(ex + ns)
    return -((pos.double() * lp).sum(1) / pos.double().sum(1).clamp_min(1.0)).mean()

def ema_live(pr, cn, z, y, dom):
    """閉式向量化 EMA（偏離 3）：p' = normalize(p·m^k + (1−m)·Σ m^(k−j) z_j)，帶梯度。"""
    pr2, cn2 = pr.clone(), cn.clone()
    for c in range(NC):
        s = (y == c).nonzero(as_tuple=True)[0]
        if len(s) == 0: continue
        k = len(s); w = PROTO_M ** torch.arange(k - 1, -1, -1, device=dev, dtype=torch.float64)
        acc = (1 - PROTO_M) * (w.unsqueeze(1) * z[s]).sum(0)
        base = pr[c, dom] * (PROTO_M ** k) if cn[c, dom] > 0 else 0.0
        pr2 = pr2.clone(); pr2[c, dom] = nrm(base + acc) if cn[c, dom] > 0 else nrm(z[s].mean(0))
        cn2 = cn2.clone(); cn2[c, dom] = cn[c, dom] + k
    return pr2, cn2

ARMS = [("arm1 現況（6格）", 6, "std", True), ("arm2 完整2b（18格・3正）", 18, "std", True),
        ("arm3 設計A（18格・1正・2忽略）", 18, "A", True),
        ("arm4 comp-only（錨點）", 6, "std", False)]
REPS = [0, 3, 6]
res = {a[0]: {"pt": [], "pl": [], "dp": [], "dk": []} for a in ARMS}
print()
print("=" * 96)
print(f"四臂離線微調（節點 {REPS}、{STEPS} 步、lr={LR} momentum={MOM} batch={BS}）")
print("=" * 96)
for name, nslot, mode, use_disp in ARMS:
    for i in REPS:
        W0, pr0, cn0 = BASE[i]
        W = [w.clone().requires_grad_(True) for w in W0]
        pr, cn = pr0.clone(), cn0.clone()
        dom = int((cn0 > 0).double().argmax(1)[0].item())
        opt = torch.optim.SGD(W, lr=LR, momentum=MOM)
        hs, ys = SRC_H[i], SRC_Y[i]
        keep = ys < NC
        hs, ys = hs[keep], ys[keep]
        g = torch.Generator(device="cpu").manual_seed(2026)       # 四臂共用抽樣順序
        # 評估起點（person 被拉近多少的基準）
        with torch.no_grad():
            zt = proj(W, TGT_H[i]); Cs = class_centers(pr, cn)
            s0 = detection_score(zt, Cs)
            a0p, a0k = float(s0[TGT_Y[i] == UNK].mean()), float(s0[TGT_Y[i] != UNK].mean())
        for st in range(STEPS):
            idx = torch.randint(0, len(ys), (BS,), generator=g).to(dev)
            z = proj(W, hs[idx]); y = ys[idx]
            prl, cnl = ema_live(pr, cn, z, y, dom)
            if nslot == 6:
                flat = nrm(prl[:, dom, :]); pcls = torch.arange(NC, device=dev)
                psty = torch.full((NC,), dom, device=dev)
            else:
                cols = [prl[:, dom, :] if d == dom else P3[d] for d in range(3)]
                flat = nrm(torch.stack(cols, 1).reshape(NC * 3, -1))
                pcls = torch.arange(NC, device=dev).repeat_interleave(3)
                psty = torch.tensor([0, 1, 2], device=dev).repeat(NC)
            loss = RC * comp(z, y, flat, pcls, psty, dom, mode)
            if use_disp:
                cl = cnl if nslot == 6 else torch.ones_like(cnl)
                loss = loss + RD * __import__("dood.prototype", fromlist=["disp_loss"]).disp_loss(
                    class_centers(prl if nslot == 6 else
                                  torch.stack([prl[:, dom, :] if d == dom else P3[d]
                                               for d in range(3)], 1), cl), cl, T)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pr, cn = prl.detach(), cnl.detach()
        with torch.no_grad():
            zt = proj(W, TGT_H[i]); y = TGT_Y[i]
            Cs = class_centers(pr[:, dom, :].unsqueeze(1), cn[:, dom:dom+1])   # 固定讀出：自有欄
            s = detection_score(zt, Cs); P = residual_projector(Cs)
            rs = residual_score(zt, P)
            m3, m2 = y == UNK, y != UNK
            res[name]["pt"].append(auroc(s[m3], s[m2]))
            res[name]["pl"].append(auroc(rs[m3], rs[m2]))
            res[name]["dp"].append(a0p - float(s[m3].mean()))   # person 被拉近多少（正＝拉近）
            res[name]["dk"].append(a0k - float(s[m2].mean()))   # 已知被拉近多少
    r = res[name]
    print(f"  {name:<30} 點讀出 {np.mean(r['pt']):.4f}   面讀出 {np.mean(r['pl']):.4f}   "
          f"person 被拉近 {np.mean(r['dp']):+6.2f}°  已知 {np.mean(r['dk']):+6.2f}°  "
          f"差 {np.mean(r['dp'])-np.mean(r['dk']):+6.2f}°")

b = res["arm1 現況（6格）"]
print("\n" + "=" * 96)
print("驗證2 解析度：arm1（目標沒變）漂移了多少？")
print("=" * 96)
s0pt = np.mean([pt0[i] for i in REPS]); s0pl = np.mean([pl0[i] for i in REPS])
print(f"  點讀出 {s0pt:.4f} → {np.mean(b['pt']):.4f}（漂移 {np.mean(b['pt'])-s0pt:+.4f}）")
print(f"  面讀出 {s0pl:.4f} → {np.mean(b['pl']):.4f}（漂移 {np.mean(b['pl'])-s0pl:+.4f}）")
print("\n" + "=" * 96)
print("驗證3 方向錨：arm4（disp 關）必須比 arm1 差（真實從頭訓練 −0.0318±0.0063）")
print("=" * 96)
d4 = np.mean(res["arm4 comp-only（錨點）"]["pt"]) - np.mean(b["pt"])
print(f"  arm4 − arm1（點讀出）＝ {d4:+.4f}   {'✅ 方向對' if d4 < 0 else '🚨 方向相反 ⇒ 探針不可信'}")
print("\n" + "=" * 96)
print("要測的兩臂（相對 arm1）")
print("=" * 96)
for name in ["arm2 完整2b（18格・3正）", "arm3 設計A（18格・1正・2忽略）"]:
    r = res[name]
    print(f"  {name:<30} 點 {np.mean(r['pt'])-np.mean(b['pt']):+.4f}   "
          f"面 {np.mean(r['pl'])-np.mean(b['pl']):+.4f}")
print("=" * 96)
