"""驗證 style_vs_content_decomposition.py 判準所依賴的前提：「畫風不挑類別」。

若前提為真：內容完全固定、只換畫風所造成的位移，應該以「共同成分」為主。
若前提為假：共同/類別專屬的分法就不能拿來分辨畫風與內容 ⇒ 實驗二的判讀必須撤回。

四組量測：
  A 純畫風對照：同一批 cartoon 影像做確定性 AdaIN（內容逐像素相同、只換 channel 統計量）
  B 來源域之間：art↔photo↔sketch 的兩兩位移（都在聯邦內、都被訓練過）
  C 類內散布：來源域 vs cartoon 在自己類別均值周圍的角度散布
  D 逐節點階梯（檢查 9 節點是否一致，而非被某一組拉動）

★ 2026-08-21 擴充（dany 同意）：**同時在兩個空間量**，一次 forward 兩份輸出、不重跑。
  「128維投影」＝ project() 之後（被 `L_disp` 撐開到類間 98.23°）
  「512維骨幹」＝ project() 之前的 pooled 特徵（類間僅 49.86°、四臂幾乎沒動）
  問題：128 維測到「純畫風位移也有 48% 是類別專屬」，這是畫風的本性，
        還是投影層把原本一致的位移打散了？⇒ 比 512 維就知道。
  ⚠️ confound 控制：中心分得越開、若位移是沿著各自中心的徑向，方向自動越分散。
     ⇒ 另把位移拆成「徑向」（沿自己類別中心方向）與「切向」（垂直）分別看共走比例。
  ⚠️ 兩空間都先做 L2 正規化（512 維原本沒有），否則共走比例不可比。
  ⚠️ 兩空間的類別中心一律用「該節點來源域的類別平均方向」（512 維沒有原型 buffer）；
     128 維另附「用訓練原型」的版本作 sanity check（＝原腳本數字）。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
from style_transforms import adain

PACS = ["art_painting", "cartoon", "photo", "sketch"]
# ★ 2026-08-21：可用環境變數 DESC 換臂（不設＝1a-fix，向後相容）
DESC = os.environ.get("DESC", "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix")
CK = "exp_result_" + DESC
leave = "cartoon"; UNK = 6; N = 9; NCLS = 6
DEG = 57.29577951308232
avail = [d for d in PACS if d != leave]
per = N // len(avail)
tgt_ld = TD.load_pacs_test_data("../datasets/", leave, 64, 4)[0]
src_ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in avail}


@torch.no_grad()
def dom_stats(bb, ld):
    acc = {k: [[], []] for k in ("layer1", "layer2", "layer3")}
    for b in ld:
        d, _, _ = util.unpack_batch(b)
        F = bb.extract_features_to_layer3(d.to("cuda"))
        for k in acc:
            f = F[k]; B, C_, H, W = f.shape; fl = f.view(B, C_, -1)
            acc[k][0].append(fl.mean(2).cpu()); acc[k][1].append(fl.std(2).cpu())
    return {k: (torch.cat(v[0]).mean(0).cuda(), torch.cat(v[1]).mean(0).cuda()) for k, v in acc.items()}


@torch.no_grad()
def collect(bb, ld, st=None):
    V, Z, Y = [], [], []
    for b in ld:
        d, y, _ = util.unpack_batch(b)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(d.to("cuda")))))
        h = bb.backbone.layer1(h)
        if st: h = adain(h, *st["layer1"])
        h = bb.backbone.layer2(h)
        if st: h = adain(h, *st["layer2"])
        h = bb.backbone.layer3(h)
        if st: h = adain(h, *st["layer3"])
        _, vec = bb.forward_from_layer3(h)
        V.append(vec.cpu().numpy())
        Z.append(bb.project(vec).cpu().numpy()); Y.append(np.asarray(y).flatten())
    # ★ 兩空間都 L2 正規化（project 內已含；512 維在此補上）⇒ 共走比例才可比
    V = np.concatenate(V); Z = np.concatenate(Z); Y = np.concatenate(Y)
    return V / np.linalg.norm(V, axis=1, keepdims=True), Z, Y


def common_frac(dvecs):
    dbar = dvecs.mean(0)
    return float((dbar ** 2).sum() / (dvecs ** 2).sum(1).mean())


def disp(Za, Ya, Zb, Yb):
    """逐類別位移向量 [NCLS, D]（b − a）"""
    return np.stack([Zb[Yb == c].mean(0) - Za[Ya == c].mean(0) for c in range(NCLS)])


def scatter_deg(Z, Y):
    """類內角度散布：每個樣本與自己類別均值方向的夾角（度），跨類別平均"""
    out = []
    for c in range(NCLS):
        m = Y == c
        if m.sum() < 2: continue
        mu = Z[m].mean(0); mu = mu / np.linalg.norm(mu)
        zn = Z[m] / np.linalg.norm(Z[m], axis=1, keepdims=True)
        out.append(np.arccos(np.clip(zn @ mu, -1 + 1e-7, 1 - 1e-7)).mean() * DEG)
    return float(np.mean(out))


def ang_own(Z, Y, C):
    m = Y != UNK; Z, Y = Z[m], Y[m].astype(int)
    zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return np.arccos(np.clip((zn @ C.T)[np.arange(len(Y)), Y], -1 + 1e-7, 1 - 1e-7)).mean() * DEG


nrm1 = lambda v: v / np.linalg.norm(v)


def src_centers(X, Y):
    """該空間裡「來源域各類別的平均方向」[NCLS, D]（512 維沒有原型 buffer，兩空間統一用這個）"""
    return np.stack([nrm1(X[Y == c].mean(0)) for c in range(NCLS)])


def inter_class_deg(U):
    """類別中心兩兩夾角平均（度）＝ 這個空間被撐開的程度"""
    iu = np.triu_indices(len(U), 1)
    return float(np.arccos(np.clip(U @ U.T, -1 + 1e-7, 1 - 1e-7))[iu].mean() * DEG)


def radial_tangential(dvecs, U):
    """★ confound 控制：把位移拆成「沿著自己類別中心的方向」（徑向）與「垂直於它」（切向）。

    若位移主要是徑向的，中心分得越開、位移方向就自動越分散 ⇒ 共走比例低是幾何造成的假象。
    回傳 (徑向能量佔比, 徑向共走比例, 切向共走比例)
    """
    proj = (dvecs * U).sum(1, keepdims=True) * U
    tang = dvecs - proj
    e_r = float((proj ** 2).sum(1).mean()); e_t = float((tang ** 2).sum(1).mean())
    return e_r / (e_r + e_t), common_frac(proj), common_frac(tang)


def noise_floor(X, Y):
    """雜訊地板：同一域內偶/奇索引兩半的類別均值差（沒有真實位移）⇒ 量法有無系統性偏差"""
    nv = []
    for c in range(NCLS):
        idx = np.where(Y == c)[0]
        if len(idx) < 4: continue
        nv.append(X[idx[0::2]].mean(0) - X[idx[1::2]].mean(0))
    return common_frac(np.stack(nv))


SPACES = ["128維投影", "512維骨幹"]
KEYS = ("A_FRAC A_NORM REAL_FRAC REAL_NORM B_FRAC B_NORM SC_SRC SC_TGT SC_ADAIN NOISE_FRAC "
        "INTER RAD_E RAD_F TAN_F A_RAD_E A_RAD_F A_TAN_F ANG_SRC ANG_TGT ANG_OTH").split()
R = {sp: {k: [] for k in KEYS} for sp in SPACES}
PROTO_ANG = []          # 128 維、用「訓練原型」當中心的階梯（＝原腳本數字，sanity check）

for i in range(N):
    bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
    Cp = class_centers(bb.prototypes, bb.proto_count).cpu().numpy()
    Cp = Cp / np.linalg.norm(Cp, axis=1, keepdims=True)
    own_d = avail[min(i // per, len(avail) - 1)]
    other_d = [d for d in avail if d != own_d]

    Vs, Zs, Ys = collect(bb, src_ld[own_d])
    Vt, Zt, Yt = collect(bb, tgt_ld)
    st_src = dom_stats(bb, src_ld[own_d])
    Va, Za, Ya = collect(bb, tgt_ld, st_src)      # cartoon ＋來源域統計量（內容逐像素相同）
    OTH = [collect(bb, src_ld[d_]) for d_ in other_d]

    # 只留已知類別（person 不參與位移分解）
    ms, mt, ma = Ys != UNK, Yt != UNK, Ya != UNK
    Vs, Zs, Ys = Vs[ms], Zs[ms], Ys[ms].astype(int)
    Vt, Zt, Yt = Vt[mt], Zt[mt], Yt[mt].astype(int)
    Va, Za, Ya = Va[ma], Za[ma], Ya[ma].astype(int)
    OTH = [(V[Y != UNK], Z[Y != UNK], Y[Y != UNK].astype(int)) for V, Z, Y in OTH]

    # 128 維、用訓練原型當中心的階梯（原腳本口徑）
    PROTO_ANG.append((own_d, ang_own(Zs, Ys, Cp),
                      float(np.mean([ang_own(Z, Y, Cp) for _, Z, Y in OTH])),
                      ang_own(Zt, Yt, Cp)))

    for sp, Xs, Xt, Xa, XO in [("128維投影", Zs, Zt, Za, [o[1] for o in OTH]),
                               ("512維骨幹", Vs, Vt, Va, [o[0] for o in OTH])]:
        r = R[sp]
        U = src_centers(Xs, Ys)                    # 兩空間統一：來源域類別平均方向
        r["INTER"].append(inter_class_deg(U))
        r["ANG_SRC"].append(ang_own(Xs, Ys, U)); r["ANG_TGT"].append(ang_own(Xt, Yt, U))
        r["ANG_OTH"].append(float(np.mean([ang_own(X, Y, U) for X, (_, _, Y) in zip(XO, OTH)])))

        dA = disp(Xt, Yt, Xa, Ya)                  # A 純畫風（內容逐像素固定）
        r["A_FRAC"].append(common_frac(dA)); r["A_NORM"].append(np.linalg.norm(dA, axis=1).mean())
        e, fr, ft = radial_tangential(dA, U)
        r["A_RAD_E"].append(e); r["A_RAD_F"].append(fr); r["A_TAN_F"].append(ft)

        dR = disp(Xs, Ys, Xt, Yt)                  # 真實跨域（畫風＋內容都變）
        r["REAL_FRAC"].append(common_frac(dR)); r["REAL_NORM"].append(np.linalg.norm(dR, axis=1).mean())
        e, fr, ft = radial_tangential(dR, U)
        r["RAD_E"].append(e); r["RAD_F"].append(fr); r["TAN_F"].append(ft)

        r["NOISE_FRAC"].append(noise_floor(Xt, Yt))

        for X, (_, _, Y) in zip(XO, OTH):          # B 來源域之間（兩端都在聯邦內）
            dB = disp(Xs, Ys, X, Y)
            r["B_FRAC"].append(common_frac(dB)); r["B_NORM"].append(np.linalg.norm(dB, axis=1).mean())

        r["SC_SRC"].append(scatter_deg(Xs, Ys))    # C 類內散布
        r["SC_TGT"].append(scatter_deg(Xt, Yt)); r["SC_ADAIN"].append(scatter_deg(Xa, Ya))

    del bb
    print(f"  node_{i} ({own_d}) 完成", flush=True)

m = lambda x: float(np.mean(x))
W = 92
line = lambda: print("=" * W)


def row(label, key, fmt="{:.4f}", note=""):
    a, b = m(R["128維投影"][key]), m(R["512維骨幹"][key])
    print(f"  {label:<34}{fmt.format(a):>12}{fmt.format(b):>14}   {note}")


print("\n" + "=" * W)
print("§0 自檢（必須先過）")
line()
print(f"  無資訊基準（6 個彼此無關的向量）= {1/NCLS:.4f}   ← 共走比例的零點")
row("雜訊地板（同域偶/奇兩半）", "NOISE_FRAC", note="應 ≈ 0.1667；偏高=量法有偏差")
print()
print(f"  {'':<34}{'128維投影':>12}{'512維骨幹':>14}   ← 對照 0820b §1.1/§1.2（1a-fix）")
row("類別中心兩兩夾角", "INTER", "{:.2f}°", "0820b：128=98.23° / 512=49.86°")
row("類內散布 ①來源域", "SC_SRC", "{:.2f}°", "0820b：128=31.21° / 512=28.26°")
row("類內散布 ②cartoon", "SC_TGT", "{:.2f}°", "0820b：128=40.78° / 512=30.26°")
print()
print("  128 維、用【訓練原型】當中心的三級階梯（原腳本口徑，對照 0819b §3 原樣列）")
print(f"    {'node':<5}{'來源域':<14}{'①自己':>9}{'★其他':>9}{'②cartoon':>11}{'①→★':>9}{'★→②':>9}")
for i, (d_, a, b, c_) in enumerate(PROTO_ANG):
    print(f"    {i:<5}{d_:<14}{a:>8.2f}°{b:>8.2f}°{c_:>10.2f}°{b-a:>8.2f}°{c_-b:>8.2f}°")
pa = np.array([[a, b, c_] for _, a, b, c_ in PROTO_ANG]).mean(0)
print(f"    {'平均':<19}{pa[0]:>8.2f}°{pa[1]:>8.2f}°{pa[2]:>10.2f}°{pa[1]-pa[0]:>8.2f}°{pa[2]-pa[1]:>8.2f}°"
      f"   ← 0819 原樣：29.63 / 44.28 / 53.57")

print("\n" + "=" * W)
print("★★ 主結果：位移的「六類共走比例」——投影前 vs 投影後")
line()
print(f"  {'':<34}{'128維投影':>12}{'512維骨幹':>14}")
row("A 純畫風位移（內容逐像素固定）", "A_FRAC", note="← 主判準（128 基準 0.5220）")
row("　　位移長度", "A_NORM")
row("B 真實跨域位移（畫風＋內容）", "REAL_FRAC", note="（128 基準 0.3098）")
row("　　位移長度", "REAL_NORM")
row("C 來源域之間的位移", "B_FRAC", note="（128 基準 0.4216）")
print(f"  {'無資訊基準':<34}{1/NCLS:>12.4f}{1/NCLS:>14.4f}")

print("\n" + "=" * W)
print("★ confound 控制：徑向（沿自己類別中心）vs 切向（垂直）")
line()
print("  若共走比例的差異只是「中心分得開⇒位移自動分散」，會表現為【徑向佔比高、切向共走比例兩空間相近】")
print(f"  {'':<34}{'128維投影':>12}{'512維骨幹':>14}")
row("A 純畫風：徑向能量佔比", "A_RAD_E")
row("A 純畫風：徑向共走比例", "A_RAD_F")
row("A 純畫風：切向共走比例", "A_TAN_F", note="← 排除幾何後的真訊號")
print()
row("B 真實跨域：徑向能量佔比", "RAD_E")
row("B 真實跨域：徑向共走比例", "RAD_F")
row("B 真實跨域：切向共走比例", "TAN_F", note="← 排除幾何後的真訊號")

print("\n" + "=" * W)
print("D 類內散布與 AdaIN 的修復力（兩空間）")
line()
print(f"  {'':<34}{'128維投影':>12}{'512維骨幹':>14}")
row("①來源域", "SC_SRC", "{:.2f}°")
row("②cartoon", "SC_TGT", "{:.2f}°")
row("②＋來源域統計量（AdaIN）", "SC_ADAIN", "{:.2f}°")
for sp in SPACES:
    a, b, c_ = m(R[sp]["SC_SRC"]), m(R[sp]["SC_TGT"]), m(R[sp]["SC_ADAIN"])
    print(f"  {sp}：膨脹 {b-a:+.2f}°、AdaIN 修掉 {b-c_:.2f}°（{(b-c_)/max(b-a,1e-9)*100:.0f}%）")

print("\n" + "=" * W)
print("E 到自己類別中心的角度（中心＝來源域類別平均方向，兩空間同口徑）")
line()
print(f"  {'':<34}{'128維投影':>12}{'512維骨幹':>14}")
row("①本地來源域", "ANG_SRC", "{:.2f}°")
row("★其他來源域", "ANG_OTH", "{:.2f}°")
row("②cartoon", "ANG_TGT", "{:.2f}°")

print("\n" + "=" * W)
print("判準（事前寫死）")
line()
a128, a512 = m(R["128維投影"]["A_FRAC"]), m(R["512維骨幹"]["A_FRAC"])
t128, t512 = m(R["128維投影"]["A_TAN_F"]), m(R["512維骨幹"]["A_TAN_F"])
print(f"  純畫風共走比例：128 維 {a128:.4f} → 512 維 {a512:.4f}（差 {a512-a128:+.4f}）")
print(f"  其中切向部分  ：128 維 {t128:.4f} → 512 維 {t512:.4f}（差 {t512-t128:+.4f}）")
if a512 > a128 + 0.10 and t512 > t128 + 0.05:
    print("  ⇒ ★ 情況一：位移在骨幹裡明顯更一致，且不是幾何假象 ⇒ 投影層把它打散")
    print("     ⇒ 位置那一半可做「保護共模性」")
elif a512 > a128 + 0.10:
    print("  ⇒ ⚠️ 骨幹更一致，但優勢主要在徑向 ⇒ 可能是「中心分得開」的幾何假象，歸因不成立")
else:
    print("  ⇒ ★ 情況二：骨幹空間也差不多 ⇒ 畫風位移本質上就是類別專屬")
    print("     ⇒ 位置這一半結構性受阻（推論時不知道類別），主攻收緊那一半")
