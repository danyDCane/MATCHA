"""獨立複核 matcha-fb 0930 的九項原型幾何量測（2026-10-01，matcha-02 寫）。

為什麼要有這一支：`scripts/posthoc/proto_same_vs_cross_style_angles.py` 是 fb 寫的，
那批數字會拿去判「階段 2b 跨節點原型聚合能不能單獨當第二貢獻」。依 TaskBoard 的
「進真相區要複核與錨點」規則，會下判斷的數字要獨立管線複核 ⇒ **本檔刻意不 import、
不參考 fb 那支**，只讀同一批原始資料，自己從頭算。

⚠️ 與 fb 那支的分工（不可互相取代）：
    fb 的      ＝ 原始量測，含它自己的 proxy 版 comp_loss
    本檔       ＝ 獨立重算，**直接呼叫 `dood.prototype` 的真貨**，並逐項對 fb 的值

資料來源（兩個都要）：
    logs/prototype_probe/0826_features_full.npz          `n{i}_C` [6,128]、`n{i}_src_z`/`src_y`
    exp_result_..._seed2026_topo1234_fix/*_node_{i}_*.pth  `backbone_state['prototypes'|'proto_count']`

錨點（必須逐位重現，不然本檔的結論作廢）：
    0829_why_18_prototypes.log 環節1：11.51° / 101.37° / 8.8×（畫風平均後的 18 原型）
    checkpoint 的原型欄 vs npz 的 n{i}_C：應為 0.0000°
"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dood.prototype import comp_loss, style_loss, class_centers   # ★ 真貨，不自己重寫

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NPZ  = os.path.join(ROOT, 'logs/prototype_probe/0826_features_full.npz')
TAG  = 'v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix'
RUN  = os.path.join(ROOT, 'exp_result_' + TAG)
NODE_STYLE = [0,0,0, 1,1,1, 2,2,2]          # 0=art_painting 1=photo 2=sketch（由 0826_probe_info.log 確認）
STYLE_NAME = ['art_painting', 'photo', 'sketch']
NCLS, NDOM, NPROT = 6, 3, 9
TEMP = 0.1                                   # train.py 的 --proto_temp 預設，該 run 未改

def ang(a, b):
    """兩個單位向量的夾角（度）。輸入不必預先正規化。"""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))

def chord(theta_deg):
    """單位球上的弦長 d = 2 sin(θ/2)。角度不可直接相加減，弦長平方才可以。"""
    return 2.0 * np.sin(np.radians(theta_deg) / 2.0)

def unchord(d):
    return float(np.degrees(2.0 * np.arcsin(np.clip(d / 2.0, -1.0, 1.0))))

def load_ckpt_protos(epoch='final'):
    """回傳 [9, 6, 3, 128] 的原型與 [9, 6, 3] 的計數（float64）。"""
    P = np.zeros((NPROT, NCLS, NDOM, 128)); K = np.zeros((NPROT, NCLS, NDOM))
    for i in range(NPROT):
        suf = 'final' if epoch == 'final' else f'epoch_{epoch}'
        ck = torch.load(os.path.join(RUN, f'{TAG}_node_{i}_{suf}.pth'),
                        map_location='cpu', weights_only=False)
        bs = ck['backbone_state']
        P[i] = bs['prototypes'].double().numpy(); K[i] = bs['proto_count'].double().numpy()
    return P, K

def style_avg_protos(P):
    """階段 2b 的目標形狀：[6, 3, 128]，第 s 欄 = 畫風 s 那三個節點自有欄的平均（再正規化）。
    這與 0829 環節1 的「畫風平均後的 18 原型」是同一個構造 ⇒ 錨點可比。"""
    out = np.zeros((NCLS, NDOM, 128))
    for s in range(NDOM):
        nodes = [i for i in range(NPROT) if NODE_STYLE[i] == s]
        m = np.mean([P[i][:, s, :] for i in nodes], axis=0)       # [6,128]
        out[:, s, :] = m / np.linalg.norm(m, axis=1, keepdims=True)
    return out

def own_col(P, i):
    return P[i][:, NODE_STYLE[i], :]                              # [6,128] 該節點自有欄

# ─────────────────────────────────────────────────────────────────────────────
print('=' * 86)
print('§0 錨點自檢')
print('=' * 86)
d = np.load(NPZ)
P_fin, K_fin = load_ckpt_protos('final')
mx = max(ang(own_col(P_fin, i)[c], d[f'n{i}_C'][c].astype(np.float64))
         for i in range(NPROT) for c in range(NCLS))
print(f'  checkpoint 自有欄 vs npz n{{i}}_C 的最大夾角：{mx:.4f}°   （fb 說應為 0.0000°）')

PA = style_avg_protos(P_fin)                                      # 18 個原型
flat = PA.reshape(NCLS * NDOM, 128)
fcls = np.repeat(np.arange(NCLS), NDOM)
same_c = [ang(flat[a], flat[b]) for a in range(18) for b in range(a+1, 18) if fcls[a] == fcls[b]]
diff_c = [ang(flat[a], flat[b]) for a in range(18) for b in range(a+1, 18) if fcls[a] != fcls[b]]
print(f'  環節1 同類別跨畫風  平均 {np.mean(same_c):.2f}°  範圍 {min(same_c):.2f}–{max(same_c):.2f}°  (n={len(same_c)})'
      f'   ← 0829 報 11.51° / 7.20–16.17° / n=18')
print(f'  環節1 不同類別      平均 {np.mean(diff_c):.2f}°  最小 {min(diff_c):.2f}°  (n={len(diff_c)})'
      f'   ← 0829 報 101.37° / 98.41° / n=135')
print(f'  環節1 比值          {np.mean(diff_c)/np.mean(same_c):.1f}×                 ← 0829 報 8.8×')

# ─────────────────────────────────────────────────────────────────────────────
def geom(P):
    """第 1/2/3/4/5 項。回傳 dict。"""
    s1 = [ang(own_col(P, i)[c], own_col(P, j)[c])
          for c in range(NCLS) for i in range(NPROT) for j in range(i+1, NPROT)
          if NODE_STYLE[i] == NODE_STYLE[j]]
    s2 = [ang(own_col(P, i)[c], own_col(P, j)[c])
          for c in range(NCLS) for i in range(NPROT) for j in range(i+1, NPROT)
          if NODE_STYLE[i] != NODE_STYLE[j]]
    s3 = [ang(own_col(P, i)[a], own_col(P, i)[b])
          for i in range(NPROT) for a in range(NCLS) for b in range(a+1, NCLS)]
    # 弦長平方分解：同畫風不同節點 ⇒ 只有兩份節點雜訊；跨畫風 ⇒ 再多一份畫風間距
    m1 = np.mean(chord(np.array(s1)) ** 2)
    m2 = np.mean(chord(np.array(s2)) ** 2)
    return dict(s1=s1, s2=s2, s3=s3,
                node=unchord(np.sqrt(m1 / 2.0)),                  # 單一節點相對該畫風真值的雜訊
                style=unchord(np.sqrt(max(m2 - m1, 0.0))))        # 扣掉節點雜訊後的純畫風間距

g = geom(P_fin)
print()
print('=' * 86)
print('§1 九項複核（final checkpoint）')
print('=' * 86)
print(f'  1  同類別・同畫風・不同節點   {np.mean(g["s1"]):7.2f}°  範圍 {min(g["s1"]):.2f}–{max(g["s1"]):.2f}°  n={len(g["s1"])}   ← fb  7.08° (4.25–11.67, n=54)')
print(f'  2  同類別・跨畫風・不同節點   {np.mean(g["s2"]):7.2f}°  範圍 {min(g["s2"]):.2f}–{max(g["s2"]):.2f}°  n={len(g["s2"])}  ← fb 12.95° (7.45–18.15, n=162)')
print(f'  3  不同類別（同節點內）       {np.mean(g["s3"]):7.2f}°  最小 {min(g["s3"]):.2f}°  n={len(g["s3"])}        ← fb 101.44°')
print(f'  4  純節點雜訊 σ_node          {g["node"]:7.2f}°                                 ← fb  5.12°')
print(f'  5  純畫風間距                 {g["style"]:7.2f}°                                 ← fb 11.03°')

# ── 第 6 項：類別中心 1 欄 → 3 欄 ───────────────────────────────────────────
print()
K1 = torch.from_numpy(K_fin[0] > 0).double()                      # 階段 1 遮罩（只有自有欄）
K3 = torch.ones(NCLS, NDOM, dtype=torch.float64)                  # 階段 2b 遮罩（三欄都有）
shift = []
for i in range(NPROT):
    c1 = class_centers(torch.from_numpy(P_fin[i]), torch.from_numpy(K_fin[i])).numpy()
    c3 = class_centers(torch.from_numpy(PA), K3).numpy()
    shift += [ang(c1[c], c3[c]) for c in range(NCLS)]
print(f'  6  類別中心 1 欄→3 欄 位移     {np.mean(shift):7.2f}°  範圍 {min(shift):.2f}–{max(shift):.2f}°  n={len(shift)}  ← fb  7.88°')

# ── 第 7、8 項：損失梯度 ────────────────────────────────────────────────────
# ⚠️ src_y 含 person(=6)，但 --excl_person 訓練從未看過 ⇒ 必須先濾掉 y<6
rows7, rows8 = [], []
for i in range(NPROT):
    z = torch.from_numpy(d[f'n{i}_src_z'].astype(np.float64))
    y = torch.from_numpy(d[f'n{i}_src_y'].astype(np.int64))
    keep = y < NCLS
    z, y = z[keep], y[keep]
    z = z / z.norm(dim=1, keepdim=True)

    def grad_of(fn):
        zz = z.clone().requires_grad_(True)
        fn(zz).backward()
        return zz.grad.detach().numpy()

    p1 = torch.from_numpy(P_fin[i]); k1 = torch.from_numpy(K_fin[i])
    p3 = torch.from_numpy(PA);       k3 = K3
    g1 = grad_of(lambda zz: comp_loss(zz, y, p1, k1, TEMP))
    g3 = grad_of(lambda zz: comp_loss(zz, y, p3, k3, TEMP))
    rows7.append((ang(g1.ravel(), g3.ravel()),
                  np.linalg.norm(g3) / np.linalg.norm(g1)))

    c1 = class_centers(p1, k1); c3 = class_centers(p3, k3)
    s1 = grad_of(lambda zz: style_loss(zz, y, c1))
    s3 = grad_of(lambda zz: style_loss(zz, y, c3))
    rows8.append((ang(s1.ravel(), s3.ravel()),
                  np.linalg.norm(s3) / np.linalg.norm(s1)))

a7 = np.array(rows7); a8 = np.array(rows8)
print(f'  7  comp_loss  階段1→2b         {a7[:,0].mean():7.2f}° ／ {a7[:,1].mean():.2f}×'
      f'   逐節點角 {a7[:,0].min():.2f}–{a7[:,0].max():.2f}°  比 {a7[:,1].min():.2f}–{a7[:,1].max():.2f}×   ← fb  8.55° ／ 1.72×')
print(f'  8  style_loss 階段1→2b         {a8[:,0].mean():7.2f}° ／ {a8[:,1].mean():.2f}×'
      f'   逐節點角 {a8[:,0].min():.2f}–{a8[:,0].max():.2f}°  比 {a8[:,1].min():.2f}–{a8[:,1].max():.2f}×   ← fb  7.81° ／ 1.00×')

# ── 第 9 項：軌跡 ───────────────────────────────────────────────────────────
print()
print('=' * 86)
print('§2 第 9 項：軌跡 ep50 → final')
print('=' * 86)
print(f'  {"epoch":>7} {"①同畫風":>10} {"②跨畫風":>10} {"④σ_node":>10} {"⑤純畫風":>10}')
for ep in [50, 100, 150, 200, 'final']:
    Pe, _ = load_ckpt_protos(ep)
    ge = geom(Pe)
    print(f'  {str(ep):>7} {np.mean(ge["s1"]):9.2f}° {np.mean(ge["s2"]):9.2f}° {ge["node"]:9.2f}° {ge["style"]:9.2f}°')
print('      ← fb 說「幾乎不動（純畫風 12.21→11.03°）」')
print('=' * 86)

# ─────────────────────────────────────────────────────────────────────────────
# §3 第 7 項對不上的根因：口徑 ＋ 梯度範數偏斜
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 86)
print('§3 第 7 項：四種口徑都算一次（fb 8.55° / 1.72×）')
print('=' * 86)

def prep(i):
    z = torch.from_numpy(d[f'n{i}_src_z'].astype(np.float64))
    y = torch.from_numpy(d[f'n{i}_src_y'].astype(np.int64))
    m = y < NCLS                                      # ★ person(=6) 必須濾掉：--excl_person 訓練從未看過
    z, y = z[m], y[m]
    return z / z.norm(dim=1, keepdim=True), y

def grad(z, y, p, k):
    zz = z.clone().requires_grad_(True)
    comp_loss(zz, y, torch.as_tensor(p), torch.as_tensor(k), TEMP).backward()
    return zz.grad.numpy()

def four(g1, g3, sel=None):
    """四種口徑：整矩陣夾角／整矩陣範數比／逐樣本夾角平均／逐樣本範數平均之比。"""
    if sel is not None: g1, g3 = g1[sel], g3[sel]
    n1 = np.linalg.norm(g1, axis=1); n3 = np.linalg.norm(g3, axis=1); ok = n1 > 1e-18
    cs = (g1[ok] * g3[ok]).sum(1) / (n1[ok] * n3[ok])
    return (ang(g1.ravel(), g3.ravel()), np.linalg.norm(g3) / np.linalg.norm(g1),
            np.degrees(np.arccos(np.clip(cs, -1, 1))).mean(), n3.mean() / n1.mean())

rows = []
for i in range(NPROT):
    z, y = prep(i)
    rows.append(four(grad(z, y, P_fin[i], K_fin[i]), grad(z, y, PA, K3)))
Q = np.array(rows)
print(f'  整矩陣夾角（本檔 §1 用的）     {Q[:,0].mean():7.2f}°')
print(f'  整矩陣範數比                   {Q[:,1].mean():7.2f}×')
print(f'  逐樣本夾角平均                 {Q[:,2].mean():7.2f}°   ← 對上 fb 的 8.55°')
print(f'  逐樣本範數「平均的比」         {Q[:,3].mean():7.2f}×   ← 對上 fb 的 1.72×')
print('  ⇒ fb 整項用逐樣本口徑、本檔 §1 用整矩陣口徑。兩個都算對了，算的是不同的量。')

print()
print('  為什麼兩種口徑差兩倍——梯度範數極度偏斜（comp_loss docstring 自己寫的飽和）：')
for i in [0, 3, 6]:
    z, y = prep(i); n1 = np.linalg.norm(grad(z, y, P_fin[i], K_fin[i]), axis=1)
    q = np.percentile(n1, [50, 99]); t = np.sort(n1)[::-1]
    print(f'    node{i}({STYLE_NAME[NODE_STYLE[i]]:<12}) 中位 {q[0]:.2e}  p99 {q[1]:.2e}  '
          f'p99/中位 {q[1]/q[0]:7.0f}×   最大 1% 樣本佔總平方範數 {(t[:len(t)//100]**2).sum()/(n1**2).sum()*100:5.1f}%')
print('    ⇒ 「逐樣本平均」量到的幾乎全是 ‖g‖~1e-6 的那 99%，與真正推動骨幹的量無關。')

# ─────────────────────────────────────────────────────────────────────────────
# §4 無資訊基準：階段 2b 的多出來那 12 格，換成「同樣遠、但方向隨機」會怎樣
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 86)
print('§4 無資訊基準（fb 沒建）：外來兩欄換成「距自有欄 11.03°、方向隨機」')
print('=' * 86)
rng = np.random.default_rng(2026)
RE, NU, RT, NT = [], [], [], []
for i in range(NPROT):
    z, y = prep(i); g1 = grad(z, y, P_fin[i], K_fin[i]); own = P_fin[i][:, NODE_STYLE[i], :]
    top = np.linalg.norm(g1, axis=1) >= np.percentile(np.linalg.norm(g1, axis=1), 99)
    pr = PA.copy(); pr[:, NODE_STYLE[i], :] = own                # 與基準同構：自有欄都保留原值
    g3 = grad(z, y, pr, K3); RE.append(four(g1, g3)); RT.append(four(g1, g3, top))
    for _ in range(20):
        pn = np.zeros_like(PA)
        for s in range(NDOM):
            if s == NODE_STYLE[i]: pn[:, s, :] = own; continue
            v = rng.normal(size=own.shape); v -= (v * own).sum(1, keepdims=True) * own
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            th = np.radians(11.03); mm = own * np.cos(th) + v * np.sin(th)
            pn[:, s, :] = mm / np.linalg.norm(mm, axis=1, keepdims=True)
        gn = grad(z, y, pn, K3); NU.append(four(g1, gn)); NT.append(four(g1, gn, top))
RE, NU, RT, NT = map(np.array, (RE, NU, RT, NT))
LAB = ['整矩陣夾角', '整矩陣範數比', '逐樣本夾角', '逐樣本範數比']
for nm, (a, b) in [('全部樣本', (RE, NU)), ('只看梯度最大的 1%（真正推動骨幹的）', (RT, NT))]:
    print(f'  ── {nm} ──')
    for j, L in enumerate(LAB):
        u = '°' if '夾角' in L else '×'
        print(f'    {L:<12} 真實 {a[:,j].mean():7.3f}{u}   隨機基準 {b[:,j].mean():7.3f} ± {b[:,j].std():.3f}{u}'
              f'   ⇒ {(a[:,j].mean()-b[:,j].mean())/b[:,j].std():+6.2f}σ')
print('  ⇒ 1.72× 就是零點（+0.10σ）：那是「原型從 6 格變 18 格、softmax 分母變了」的計數效應，')
print('    與「多出來的是不是真的跨畫風資訊」無關。唯一離零點遠的是逐樣本夾角（+18σ），')
print('    但它住在 ‖g‖~1e-6 的樣本裡。')
print('=' * 86)
