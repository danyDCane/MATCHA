"""階段 2b（跨節點原型聚合）的【必要條件】檢驗：三個畫風欄到底有沒有結構？

為什麼要補這支：`why_18_prototypes_dont_help.py` 環節1 量到「同一類、不同畫風 11.51°」，
但它比的是 **畫風平均後**的 18 個原型（同畫風三節點先被 mean 掉）⇒ **對照組沒有被量**。
沒有「同畫風、不同節點」這個對照，就分不出 11.51° 裡有多少是畫風、多少是節點雜訊。

本檔量四件事，全部用同一份 `0826_features_full.npz`（BN 平均 B、9 節點逐節點特徵）：
  §1 同畫風跨節點 vs 跨畫風跨節點（**單一變因對照**：兩組都是「不同節點、不同模型」，只差畫風）
  §2 無資訊零點（128 維隨機單位向量的夾角分佈）——高維下無關方向是 90°，不是 180°
  §3 類別中心實際會移動多少（1 欄 → 3 欄）
  §4 ★ `comp_loss` 飽和檢查：階段 2b 的「3 個正樣本」在飽和的 softmax 裡還剩多少梯度
     （2026-08-14 已有先例：往同一個 softmax 加正樣本，單元測試實測差 1e-4）

⚠️ 射程：單 fold（leave-cartoon）、單 seed、`epoch_200` 之後的終態。
⚠️ §4 的階段 2b 原型是**代理**——真實 2b 收到的是帶延遲的鄰居原型，本檔用「同畫風三節點平均」
   當上界（無延遲、完全可見）。故 §4 若在上界都沒梯度，真實情況只會更少。
"""
import numpy as np, torch, torch.nn.functional as Fn

DEG = 57.29577951308232
N, NC = 9, 6
AVAIL = ["art", "photo", "sketch"]
OWN = [AVAIL[i // 3] for i in range(N)]
F = np.load("logs/prototype_probe/0826_features_full.npz")
nrm = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)

CEN = nrm(np.stack([F[f"n{i}_C"] for i in range(N)]).astype(np.float64))   # [9,6,128] 逐節點類別中心
P3 = nrm(np.stack([CEN[[i for i in range(N) if OWN[i] == d]].mean(0)
                   for d in AVAIL]))                                        # [3,6,128] 畫風平均
P18 = P3.transpose(1, 0, 2).reshape(NC * 3, -1)                             # [18,128]
LAB18 = np.repeat(np.arange(NC), 3)
ang = lambda a, b: np.arccos(np.clip(np.dot(a, b), -1, 1)) * DEG

def stat(v, name, w=34):
    v = np.asarray(v)
    print(f"  {name:<{w}}平均 {v.mean():6.2f}°   中位 {np.median(v):6.2f}°   "
          f"範圍 {v.min():5.2f}–{v.max():6.2f}°   (n={len(v)})")

print("=" * 96)
print("§0 錨點：逐位重現 `0829_why_18_prototypes.log` 環節1（畫風平均後的 18 原型）")
print("=" * 96)
G = np.arccos(np.clip(P18 @ P18.T, -1, 1)) * DEG
a_same = [G[a, b] for a in range(18) for b in range(a + 1, 18) if LAB18[a] == LAB18[b]]
a_diff = [G[a, b] for a in range(18) for b in range(a + 1, 18) if LAB18[a] != LAB18[b]]
print(f"  同一類、不同畫風  平均 {np.mean(a_same):6.2f}°  範圍 {np.min(a_same):5.2f}–"
      f"{np.max(a_same):5.2f}°  (n={len(a_same)})    ← 0829 log 報 11.51° / 7.20–16.17° / n=18")
print(f"  不同類別          平均 {np.mean(a_diff):6.2f}°  最小 {np.min(a_diff):5.2f}°"
      f"            (n={len(a_diff)})   ← 0829 log 報 101.37° / 98.41° / n=135")
print(f"  比值 {np.mean(a_diff)/np.mean(a_same):.1f}×   ← 0829 log 報 8.8×")

print()
print("=" * 96)
print("§1 ★ 補上缺的對照：同畫風跨節點 vs 跨畫風跨節點（**都用逐節點原型、未先平均**）")
print("=" * 96)
same_sty, cross_sty, diff_cls = [], [], []
for c in range(NC):
    for i in range(N):
        for j in range(i + 1, N):
            (same_sty if OWN[i] == OWN[j] else cross_sty).append(ang(CEN[i, c], CEN[j, c]))
    for c2 in range(c + 1, NC):
        for i in range(N):
            diff_cls.append(ang(CEN[i, c], CEN[i, c2]))
stat(same_sty,  "① 同類別・同畫風・不同節點")
stat(cross_sty, "② 同類別・跨畫風・不同節點")
stat(diff_cls,  "③ 不同類別（同一節點內）")
print(f"\n  ⇒ 畫風真正貢獻的量 ＝ ② − ① ＝ {np.mean(cross_sty)-np.mean(same_sty):.2f}°"
      f"   （① 是「不同節點、不同模型」的純雜訊底線）")
print(f"  ⇒ ② 之中節點雜訊佔 {np.mean(same_sty)/np.mean(cross_sty)*100:.1f}%，畫風佔 "
      f"{(1-np.mean(same_sty)/np.mean(cross_sty))*100:.1f}%")
print(f"  ⇒ ② 相對 ③（類別間距）：{np.mean(cross_sty)/np.mean(diff_cls)*100:.1f}%")

print("\n  逐畫風拆（②：該畫風 vs 另外兩個畫風）")
for k, d in enumerate(AVAIL):
    grp = [i for i in range(N) if OWN[i] == d]
    oth = [i for i in range(N) if OWN[i] != d]
    v_in = [ang(CEN[i, c], CEN[j, c]) for c in range(NC) for i in grp for j in grp if i < j]
    v_out = [ang(CEN[i, c], CEN[j, c]) for c in range(NC) for i in grp for j in oth]
    print(f"    {d:<8} 組內 {np.mean(v_in):5.2f}°   對外 {np.mean(v_out):5.2f}°   "
          f"差 {np.mean(v_out)-np.mean(v_in):+5.2f}°")

print()
print("=" * 96)
print("§2 無資訊零點（128 維隨機單位向量；高維下無關方向 ≈ 90°）")
print("=" * 96)
rng = np.random.default_rng(2026)
R = nrm(rng.standard_normal((4000, 128)))
rv = np.arccos(np.clip((R[:2000] * R[2000:]).sum(1), -1, 1)) * DEG
print(f"  隨機兩方向  平均 {rv.mean():6.2f}°   std {rv.std():.2f}°   p1–p99 "
      f"{np.percentile(rv,1):.2f}–{np.percentile(rv,99):.2f}°")
print(f"  ⇒ ①②③ 全部遠低於 90° ⇒ 六個類別中心擠在一個遠小於整個球面的錐體裡（已知事實，非新發現）")

print()
print("=" * 96)
print("§3 類別中心實際會移動多少：現行（只有自己那欄）→ 階段 2b（三欄平均）")
print("=" * 96)
C2b = nrm(P3.mean(0))                       # [6,128] 三畫風平均後的類別中心（2b 的上界代理）
shift = [ang(CEN[i, c], C2b[c]) for i in range(N) for c in range(NC)]
stat(shift, "中心位移角（現行 → 2b）", 34)
print(f"  對照：① 同畫風節點雜訊 {np.mean(same_sty):.2f}°  ⇒ 位移是雜訊的 "
      f"{np.mean(shift)/np.mean(same_sty):.2f} 倍")

print()
print("=" * 96)
print("§4 ★ `comp_loss` 飽和檢查：階段 2b 多出的兩個正樣本，還剩多少梯度？")
print("=" * 96)
T = 0.1
print(f"  設定：temperature={T}（`train.py` 預設）｜來源域 `src_z`、**已濾掉 person(=6)**｜float64")
print(f"  {'節點':<6}{'損失 階段1':>12}{'損失 2b':>12}{'Δ損失':>12}"
      f"{'‖∂L/∂z‖ 階段1':>16}{'‖∂L/∂z‖ 2b':>14}{'比值':>8}")
print("  " + "-" * 82)
rows = []
for i in range(N):
    _y = F[f"n{i}_src_y"].astype(np.int64)
    keep = _y < NC                                   # ⚠️ 傾印含 person(=6)，但 --excl_person 訓練從未看過
    z = torch.tensor(nrm(F[f"n{i}_src_z"].astype(np.float64))[keep])
    y = torch.tensor(_y[keep])
    out = []
    for flat_np, lab_np in [(CEN[i], np.arange(NC)), (P18, LAB18)]:
        zz = z.clone().requires_grad_(True)
        flat = torch.tensor(nrm(flat_np))
        proxy = torch.tensor(lab_np)
        lg = (zz @ flat.t()) / T
        lg = lg - lg.max(dim=1, keepdim=True).values.detach()
        pos = (y.view(-1, 1) == proxy.view(1, -1)).double()
        ex = torch.exp(lg)
        neg = (ex * (1 - pos)).sum(1, keepdim=True)
        lp = lg - torch.log(ex + neg)
        loss = -((pos * lp).sum(1) / pos.sum(1).clamp_min(1.0)).mean()
        loss.backward()
        out.append((loss.item(), zz.grad.norm(dim=1).mean().item(), neg.mean().item()))
    rows.append((i, out))
    print(f"  {i:<6}{out[0][0]:>12.3e}{out[1][0]:>12.3e}{out[1][0]-out[0][0]:>+12.3e}"
          f"{out[0][1]:>16.3e}{out[1][1]:>14.3e}{out[1][1]/max(out[0][1],1e-300):>8.2f}")
L1 = np.mean([r[1][0][0] for r in rows]); L2 = np.mean([r[1][1][0] for r in rows])
G1 = np.mean([r[1][0][1] for r in rows]); G2 = np.mean([r[1][1][1] for r in rows])
NS = np.mean([r[1][0][2] for r in rows])
print("  " + "-" * 82)
print(f"  {'平均':<6}{L1:>12.3e}{L2:>12.3e}{L2-L1:>+12.3e}{G1:>16.3e}{G2:>14.3e}{G2/max(G1,1e-300):>8.2f}")
print(f"\n  負樣本分母 neg_sum 平均 ＝ {NS:.3e}（`comp_loss` docstring 記錄的飽和值 2.2e-04）")
print(f"  ⇒ 階段 2b 讓 `L_comp` 的梯度大小變成 {G2/max(G1,1e-300):.2f}×")

print()
print("=" * 96)
print("§5 對照：`style_loss` 在階段 2b 才活過來（它不是 softmax、不會飽和）")
print("=" * 96)
print(f"  {'節點':<6}{'L_style 階段1':>16}{'L_style 2b':>14}{'‖∂L/∂z‖ 階段1':>18}{'‖∂L/∂z‖ 2b':>14}")
print("  " + "-" * 70)
sr = []
for i in range(N):
    _y = F[f"n{i}_src_y"].astype(np.int64)
    keep = _y < NC
    z = torch.tensor(nrm(F[f"n{i}_src_z"].astype(np.float64))[keep])
    y = torch.tensor(_y[keep])
    o = []
    for cen_np in [CEN[i], C2b]:
        zz = z.clone().requires_grad_(True)
        cen = torch.tensor(nrm(cen_np))
        ls = (1.0 - (zz * cen[y]).sum(1)).mean()
        ls.backward()
        o.append((ls.item(), zz.grad.norm(dim=1).mean().item()))
    sr.append(o)
    print(f"  {i:<6}{o[0][0]:>16.4f}{o[1][0]:>14.4f}{o[0][1]:>18.3e}{o[1][1]:>14.3e}")
print("  " + "-" * 70)
print(f"  {'平均':<6}{np.mean([r[0][0] for r in sr]):>16.4f}{np.mean([r[1][0] for r in sr]):>14.4f}"
      f"{np.mean([r[0][1] for r in sr]):>18.3e}{np.mean([r[1][1] for r in sr]):>14.3e}")
print(f"\n  ⚠️ 階段1 的 L_style 與 L_comp 是同一個量（`train.py:501` 會印警告）⇒ 上表左欄只是參照，"
      f"實際從未啟用（λ_style 一路為 0）")
print("=" * 96)

print()
print("=" * 96)
print("§1b 變異數分解（用弦長 ‖a−b‖＝2sin(θ/2) 做，歐氏距離才能直接拆；角度不可相加減）")
print("=" * 96)
# 單因子變異數分析：畫風是組別、同畫風三節點是組內重複
wi, bt = [], []
for c in range(NC):
    means = []
    for k, d in enumerate(AVAIL):
        g = CEN[[i for i in range(N) if OWN[i] == d], c]        # [3,128]
        mu = g.mean(0); means.append(mu)
        wi.append(((g - mu) ** 2).sum(1).mean())                 # 組內散布
    M = np.stack(means); gm = M.mean(0)
    bt.append(((M - gm) ** 2).sum(1).mean())                     # 組間散布（含 1/3 的組內殘留）
s_w, s_b_raw = np.mean(wi), np.mean(bt)
s_b = max(s_b_raw - s_w / 3.0, 0.0)                              # 扣掉組內殘留 ⇒ 純畫風
d2a = lambda v: np.degrees(2 * np.arcsin(np.clip(np.sqrt(max(v, 0)) / 2, 0, 1)))
print(f"  組內（節點雜訊）散布 σ²_node   = {s_w:.5f}   ⇒ 典型偏移 {d2a(s_w):5.2f}°")
print(f"  組間（未扣殘留）              = {s_b_raw:.5f}")
print(f"  組間（扣掉 σ²_node/3）＝純畫風  = {s_b:.5f}   ⇒ 典型偏移 {d2a(s_b):5.2f}°")
print(f"  ⇒ 畫風 : 節點雜訊 的散布比 ＝ {s_b/s_w:.2f} : 1"
      f"   （<1 代表節點之間的差異比畫風之間還大）")
# ★ 不需要任何模型的精確拆法：弦長平方可直接相減
#   同畫風配對  E‖pi−pj‖² = 2σ²_node
#   跨畫風配對  E‖pi−pj‖² = ‖μa−μb‖² + 2σ²_node
#   ⇒ 兩者相減 ＝ 純畫風間距的平方（節點雜訊被消掉，無需假設）
ch2 = lambda deg: (2 * np.sin(np.radians(np.asarray(deg)) / 2)) ** 2
c1, c2 = ch2(same_sty).mean(), ch2(cross_sty).mean()
print(f"\n  ★ 模型無關的精確拆解（弦長平方相減，節點雜訊自動消掉）")
print(f"    同畫風配對 弦²  {c1:.5f}   ⇒ 純節點雜訊 σ²_node ＝ {c1/2:.5f}（典型 {d2a(c1/2):.2f}°）")
print(f"    跨畫風配對 弦²  {c2:.5f}")
print(f"    相減 ＝ 純畫風間距²  {c2-c1:.5f}   ⇒ **純畫風間距 ＝ {d2a(c2-c1):.2f}°**")
print(f"    ⇒ 拿掉節點雜訊後，同一類別的三個畫風原型相距 {d2a(c2-c1):.2f}°，"
      f"而不同類別相距 {np.mean(diff_cls):.1f}°")
print(f"    ⇒ 畫風位移只有類別間距的 {d2a(c2-c1)/np.mean(diff_cls)*100:.1f}%")

print()
print("=" * 96)
print("§6 ★★ 決定性檢驗：階段 2b 的梯度是【換方向】還是【只變大】？")
print("=" * 96)
print("  若 cos ≈ 1 ⇒ 2b 等於把 λ 乘上一個常數 ⇒ **是偷改超參數、不是新機制**（📌G）")
print(f"\n  {'節點':<6}{'L_comp cos':>14}{'L_comp 倍率':>14}{'L_style cos':>14}{'L_style 倍率':>14}")
print("  " + "-" * 64)
cc, cn, sc, sn = [], [], [], []
for i in range(N):
    _y = F[f"n{i}_src_y"].astype(np.int64); keep = _y < NC
    z = torch.tensor(nrm(F[f"n{i}_src_z"].astype(np.float64))[keep])
    y = torch.tensor(_y[keep])

    def gcomp(flat_np, lab_np):
        zz = z.clone().requires_grad_(True)
        flat = torch.tensor(nrm(flat_np)); proxy = torch.tensor(lab_np)
        lg = (zz @ flat.t()) / T
        lg = lg - lg.max(dim=1, keepdim=True).values.detach()
        pos = (y.view(-1, 1) == proxy.view(1, -1)).double()
        ex = torch.exp(lg); neg = (ex * (1 - pos)).sum(1, keepdim=True)
        (-((pos * (lg - torch.log(ex + neg))).sum(1) / pos.sum(1).clamp_min(1.0)).mean()).backward()
        return zz.grad.clone()

    def gstyle(cen_np):
        zz = z.clone().requires_grad_(True)
        cen = torch.tensor(nrm(cen_np))
        (1.0 - (zz * cen[y]).sum(1)).mean().backward()
        return zz.grad.clone()

    for f, args1, args2, L, Lr in [(gcomp, (CEN[i], np.arange(NC)), (P18, LAB18), cc, cn),
                                   (gstyle, (CEN[i],), (C2b,), sc, sn)]:
        g1, g2 = f(*args1), f(*args2)
        n1, n2 = g1.norm(dim=1), g2.norm(dim=1)
        ok = (n1 > 0) & (n2 > 0)
        L.append(((g1 * g2).sum(1)[ok] / (n1[ok] * n2[ok])).mean().item())
        Lr.append((n2[ok].mean() / n1[ok].mean()).item())
    print(f"  {i:<6}{cc[-1]:>14.4f}{cn[-1]:>14.2f}{sc[-1]:>14.4f}{sn[-1]:>14.2f}")
print("  " + "-" * 64)
print(f"  {'平均':<6}{np.mean(cc):>14.4f}{np.mean(cn):>14.2f}{np.mean(sc):>14.4f}{np.mean(sn):>14.2f}")
print(f"\n  L_comp  ：方向夾角 {np.degrees(np.arccos(np.clip(np.mean(cc),-1,1))):.2f}°，大小 {np.mean(cn):.2f}×")
print(f"  L_style ：方向夾角 {np.degrees(np.arccos(np.clip(np.mean(sc),-1,1))):.2f}°，大小 {np.mean(sn):.2f}×")
print("  ⚠️ 這是對【投影後特徵 z】的梯度，不是對模型參數的梯度；z 的梯度要再經投影層反傳，")
print("     方向可能被進一步壓縮或放大 ⇒ 本節只能當上界指標，不能當最終判決。")
print("=" * 96)
