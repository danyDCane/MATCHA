"""那 23.94° 裡，多少是「畫風」多少是「內容」？

實驗一（階梯）：本地訓練過的來源域 / 其他來源域（只透過模型聚合間接見過）/ cartoon（沒人見過）
             ⇒ 「其他來源域」那一級 = 預防做到極限的地板。
實驗二（分解）：位移向量拆「共同成分」（六類共有＝畫風不挑類別）vs「類別專屬成分」（＝內容挑類別）
             ＋ oracle 扣除（扣共同 / 扣逐類別）⇒ 均值平移最多能修回多少。

全 post-hoc、零重訓、final ckpt、cartoon fold。
⚠️ oracle 扣除用到 cartoon 的資料（逐類別版還用到標籤）⇒ 是天花板不是方法。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers

PACS = ["art_painting", "cartoon", "photo", "sketch"]
# ★ 2026-08-21：可用環境變數 DESC 換臂（不設＝1a-fix，向後相容）
DESC = os.environ.get("DESC", "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix")
CK = "exp_result_" + DESC
leave = "cartoon"; UNK = 6; N = 9; NCLS = 6
DEG = 57.29577951308232

avail = [d for d in PACS if d != leave]          # [art_painting, photo, sketch]
per = N // len(avail)                            # node 0-2→art, 3-5→photo, 6-8→sketch
tgt_ld = TD.load_pacs_test_data("../datasets/", leave, 64, 4)[0]
src_ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in avail}


@torch.no_grad()
def collect(bb, ld):
    """回傳 (v[512維、已 L2 正規化], z[128維、已 L2 正規化], y)。

    ★ 2026-08-21：加 512 維骨幹空間（project 之前），一次前向兩份輸出、不重跑。
      512 維在此補做 L2 正規化，否則兩空間的角度與共走比例不可比。
    """
    V, Z, Y = [], [], []
    for b in ld:
        d, y, _ = util.unpack_batch(b)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(d.to("cuda")))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        _, vec = bb.forward_from_layer3(h)
        V.append(vec.cpu().numpy())
        Z.append(bb.project(vec).cpu().numpy())
        Y.append(np.asarray(y).flatten())
    V = np.concatenate(V)
    return V / np.linalg.norm(V, axis=1, keepdims=True), np.concatenate(Z), np.concatenate(Y)


def ang_own(Z, Y, C):
    """到『自己類別』中心的角度（度）。只取已知類別。Z 不必是單位長度（會除以自身範數）。"""
    m = Y != UNK
    Z, Y = Z[m], Y[m].astype(int)
    zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    cos = (zn @ C.T)[np.arange(len(Y)), Y]
    return np.arccos(np.clip(cos, -1 + 1e-7, 1 - 1e-7)) * DEG


def ang_near(Z, Y, C):
    m = Y != UNK
    zn = Z[m] / np.linalg.norm(Z[m], axis=1, keepdims=True)
    return np.arccos(np.clip(zn @ C.T, -1 + 1e-7, 1 - 1e-7)).min(1) * DEG


def cls_mean(Z, Y, cls):
    m = (Y == cls)
    return Z[m].mean(0) if m.sum() else None


def decompose(dvecs):
    """dvecs: [NCLS, D]。回傳 (共同成分向量, 共同能量佔比, 逐類別殘差範數)"""
    dbar = dvecs.mean(0)
    e_tot = (dvecs ** 2).sum(1).mean()
    e_com = (dbar ** 2).sum()
    resid = dvecs - dbar
    return dbar, e_com / e_tot, np.linalg.norm(resid, axis=1)


# ────────────────────────────── 主迴圈 ──────────────────────────────
LAD = {"①本地訓練過的來源域": [], "★其他來源域(僅透過聚合間接見過)": [], "②cartoon(無人見過)": []}
LAD_NEAR = {k: [] for k in LAD}
NSAMP = {k: [] for k in LAD}
COMMON_FRAC, NOISE_FRAC = [], []
D_NORM, R_NORM, NOISE_NORM = [], [], []
OR = {"原樣": [], "扣共同成分": [], "扣逐類別成分(絕對上界)": []}
nrm1 = lambda v: v / np.linalg.norm(v)
SPACES2 = ("128維投影", "512維骨幹")
OR2 = {sp: {k: [] for k in ("原樣", "扣共同成分", "扣逐類別成分", "①參照(來源域)", "共走比例")}
       for sp in SPACES2}
CLS_N = []

for i in range(N):
    bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_final.pth"), 6, "cuda")
    C = class_centers(bb.prototypes, bb.proto_count).cpu().numpy()
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    own_d = avail[min(i // per, len(avail) - 1)]
    other_d = [d for d in avail if d != own_d]

    Vs, Zs, Ys = collect(bb, src_ld[own_d])
    Vt, Zt, Yt = collect(bb, tgt_ld)

    # ---------- 實驗一：三級階梯 ----------
    LAD["①本地訓練過的來源域"].append(ang_own(Zs, Ys, C).mean())
    LAD_NEAR["①本地訓練過的來源域"].append(ang_near(Zs, Ys, C).mean())
    NSAMP["①本地訓練過的來源域"].append((Ys != UNK).sum())

    oth = [collect(bb, src_ld[d]) for d in other_d]
    LAD["★其他來源域(僅透過聚合間接見過)"].append(np.mean([ang_own(Z, Y, C).mean() for _, Z, Y in oth]))
    LAD_NEAR["★其他來源域(僅透過聚合間接見過)"].append(np.mean([ang_near(Z, Y, C).mean() for _, Z, Y in oth]))
    NSAMP["★其他來源域(僅透過聚合間接見過)"].append(int(np.mean([(Y != UNK).sum() for _, _, Y in oth])))

    LAD["②cartoon(無人見過)"].append(ang_own(Zt, Yt, C).mean())
    LAD_NEAR["②cartoon(無人見過)"].append(ang_near(Zt, Yt, C).mean())
    NSAMP["②cartoon(無人見過)"].append((Yt != UNK).sum())

    # ---------- 實驗二：位移分解 ----------
    dvec, nper = [], []
    for c in range(NCLS):
        a, b = cls_mean(Zs, Ys, c), cls_mean(Zt, Yt, c)
        dvec.append(b - a); nper.append(int((Yt == c).sum()))
    dvec = np.stack(dvec); CLS_N.append(nper)
    dbar, frac, rn = decompose(dvec)
    COMMON_FRAC.append(frac); D_NORM.append(np.linalg.norm(dvec, axis=1).mean()); R_NORM.append(rn.mean())

    # 雜訊地板：同一域內偶/奇索引兩半的類別均值差（無真實位移，純取樣雜訊）
    nvec = []
    for c in range(NCLS):
        idx = np.where(Yt == c)[0]
        if len(idx) < 4: nvec.append(np.zeros_like(dbar)); continue
        nvec.append(Zt[idx[0::2]].mean(0) - Zt[idx[1::2]].mean(0))
    nvec = np.stack(nvec)
    _, nfrac, _ = decompose(nvec)
    NOISE_FRAC.append(nfrac); NOISE_NORM.append(np.linalg.norm(nvec, axis=1).mean())

    # ---------- oracle 扣除 ----------
    OR["原樣"].append(ang_own(Zt, Yt, C).mean())
    OR["扣共同成分"].append(ang_own(Zt - dbar, Yt, C).mean())
    Zc = Zt.copy()
    for c in range(NCLS):
        Zc[Yt == c] -= dvec[c]
    OR["扣逐類別成分(絕對上界)"].append(ang_own(Zc, Yt, C).mean())

    # ---------- ★ 2026-08-21 新增：雙空間、同口徑的 oracle 扣除 ----------
    # 兩空間統一用「來源域各類別平均方向」當中心（512 維沒有原型 buffer）
    for sp, Xs, Xt in [("128維投影", Zs, Zt), ("512維骨幹", Vs, Vt)]:
        U = np.stack([nrm1(Xs[Ys == c].mean(0)) for c in range(NCLS)])
        dv = np.stack([Xt[Yt == c].mean(0) - Xs[Ys == c].mean(0) for c in range(NCLS)])
        db = dv.mean(0)
        Xc = Xt.copy()
        for c in range(NCLS):
            Xc[Yt == c] -= dv[c]
        OR2[sp]["原樣"].append(ang_own(Xt, Yt, U).mean())
        OR2[sp]["扣共同成分"].append(ang_own(Xt - db, Yt, U).mean())
        OR2[sp]["扣逐類別成分"].append(ang_own(Xc, Yt, U).mean())
        OR2[sp]["①參照(來源域)"].append(ang_own(Xs, Ys, U).mean())
        OR2[sp]["共走比例"].append(float((db ** 2).sum() / (dv ** 2).sum(1).mean()))

    del bb
    print(f"  node_{i} ({own_d}) 完成", flush=True)

# ────────────────────────────── 輸出 ──────────────────────────────
def m(x): return float(np.mean(x))
W = 74
print("\n" + "=" * W)
print("§0 資料自檢")
print("=" * W)
print(f"1. 重現核對（0818 §3.4 用不同腳本量到 ①29.63° ／ ②53.57°）")
print(f"   本腳本 ①本地來源域 = {m(LAD['①本地訓練過的來源域']):.2f}°   ②cartoon = {m(LAD['②cartoon(無人見過)']):.2f}°")
print(f"2. 樣本數（node 平均、已排除 person）：" +
      "  ".join(f"{k[:6]}={m(v):.0f}" for k, v in NSAMP.items()))
print(f"   cartoon 逐類別樣本數（node 平均）：{[round(x) for x in np.mean(CLS_N,0)]}")
print(f"3. 無資訊基準（共同成分佔比）：6 個彼此無關的向量理論值 = {1/NCLS:.4f}")
print(f"   實測雜訊地板（同域偶/奇兩半的類別均值差）= {m(NOISE_FRAC):.4f}")
print(f"4. 訊噪比：真實位移範數 {m(D_NORM):.4f} ／ 雜訊範數 {m(NOISE_NORM):.4f} "
      f"= {m(D_NORM)/max(m(NOISE_NORM),1e-9):.2f}x")
print(f"   類別專屬殘差範數 {m(R_NORM):.4f} ／ 雜訊範數 {m(NOISE_NORM):.4f} "
      f"= {m(R_NORM)/max(m(NOISE_NORM),1e-9):.2f}x  ← <2x 則『類別專屬』與雜訊分不開")
print(f"5. checkpoint：全部 final、同一組 9 節點；跨級比較皆同一個 checkpoint 的同一次前向")
print(f"6. 空間：128 維投影後（＝讀出實際用的空間），非 512 維")

print("\n" + "=" * W)
print("實驗一：三級階梯（到自己類別中心 / 到最近中心，node-mean）")
print("=" * W)
print(f"{'':<36}{'到自己類別':>12}{'到最近中心':>12}")
for k in LAD:
    print(f"{k:<34}{m(LAD[k]):>12.2f}°{m(LAD_NEAR[k]):>11.2f}°")
lo, mid, hi = m(LAD['①本地訓練過的來源域']), m(LAD['★其他來源域(僅透過聚合間接見過)']), m(LAD['②cartoon(無人見過)'])
print("-" * W)
print(f"總落差 ①→② = {hi-lo:.2f}°   其中 ①→★ = {mid-lo:.2f}° ({(mid-lo)/(hi-lo)*100:.1f}%)"
      f"   ★→② = {hi-mid:.2f}° ({(hi-mid)/(hi-lo)*100:.1f}%)")
print("判準：★ 貼近 ①（≤5°）⇒ 畫風進訓練池就學得會，落差主因是『沒見過』")
print("      ★ 貼近 ②（≤5°）⇒ 見過也沒學會，問題比『未見畫風』更基本")
print("      ★ 在中間      ⇒ 該值即『預防做到極限』的地板")

print("\n" + "=" * W)
print("實驗二：位移分解 + oracle 扣除")
print("=" * W)
print(f"共同成分能量佔比 = {m(COMMON_FRAC):.4f}   （無資訊基準 {1/NCLS:.4f}、雜訊地板 {m(NOISE_FRAC):.4f}）")
print(f"逐節點：{[round(x,3) for x in COMMON_FRAC]}")
print("判準：>0.6 ⇒ 整團同向平移（較像畫風、原則上單一校正可修）")
print("      ≈0.167 ⇒ 各類別各走各的（較像內容、不該也不能消除）")
print()
for k in OR:
    print(f"  {k:<26}{m(OR[k]):>10.2f}°")
print(f"  {'（參照）①本地來源域':<26}{lo:>10.2f}°")
print("-" * W)
rec_c = (m(OR['原樣']) - m(OR['扣共同成分'])) / (m(OR['原樣']) - lo) * 100
rec_p = (m(OR['原樣']) - m(OR['扣逐類別成分(絕對上界)'])) / (m(OR['原樣']) - lo) * 100
print(f"扣共同成分 修回 {rec_c:.1f}%   ｜   扣逐類別成分 修回 {rec_p:.1f}%（任何均值平移法的絕對上界）")
print("⚠️ 兩者皆為 oracle：用到 cartoon 的資料，逐類別版還用到標籤 ⇒ 是天花板、不是方法")
print(f"⚠️ 若『扣逐類別』仍遠高於 ① ⇒ 落差不是均值平移，而是散布形狀改變，均值校正永遠修不掉")

print("\n" + "=" * W)
print("★★ 實驗二之二（2026-08-21 新增）：同一套 oracle 扣除，在【投影前】vs【投影後】")
print("=" * W)
print("  中心一律＝該節點來源域的類別平均方向（兩空間同口徑；128 維因此與上表的原型口徑略有差異）")
print(f"  {'':<26}{'128維投影':>13}{'512維骨幹':>14}")
for k in ("①參照(來源域)", "原樣", "扣共同成分", "扣逐類別成分"):
    print(f"  {k:<24}{m(OR2['128維投影'][k]):>12.2f}°{m(OR2['512維骨幹'][k]):>13.2f}°")
print("-" * W)
for sp in SPACES2:
    o = OR2[sp]; lo2 = m(o["①參照(來源域)"]); base = m(o["原樣"])
    gap = base - lo2
    rc = (base - m(o["扣共同成分"])) / max(gap, 1e-9) * 100
    rp = (base - m(o["扣逐類別成分"])) / max(gap, 1e-9) * 100
    print(f"  {sp}：總落差 {gap:.2f}°｜扣共同修回 {rc:.1f}%｜扣逐類別修回 {rp:.1f}%"
          f"｜共走比例 {m(o['共走比例']):.4f}")
print("-" * W)
print("  判讀：若 512 維的『扣共同修回%』與『共走比例』都明顯高於 128 維")
print("        ⇒ 畫風位移在骨幹裡本來較一致，是投影層把它打散 ⇒ 位置那半可做『保護共模性』")
print("        若兩空間相近 ⇒ 位移本質上就是類別專屬 ⇒ 位置那半結構性受阻")
