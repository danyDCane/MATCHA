# -*- coding: utf-8 -*-
"""階段一b 之前的預量測：3a／3b／3d 是什麼型別的量，以及 E5／E6 的門檻該怎麼定（2026-10-02）

fb 的 plan §12 要攻三點，但其中兩點**不必等新 run** —— 九節點的原型 checkpoint 已經
足以把 `C_i`、`G_i`、`G^{-i}`、`G^{同畫風,-i}` 全部建出來。

§A  3a／3b／3d 預量測：**節點間散布 vs 節點內隨時間的散布**
    ⇒ 若節點間 >> 節點內，這個訊號編碼的是「我是不是離群節點」，不是「該廣播了」。
§B  E6 的門檻：不要憑感覺定 0.5，**量現行訊號（規則 0／1）同一個統計量**當基準。
§C  E5 的「明顯弱於」：用 **block bootstrap 的 κ 標準誤**定量。

⚠️ 射程：§A 只有 5 個時點（ep50/100/150/200/final）⇒ 節點內隨時間的散布是**上界**
   （原型漂移慢，廣播間隔內的變動必然小於 epoch 尺度的變動）⇒ 本檢定對「節點間主宰」
   這個結論是**保守的**。
"""
import os, sys, numpy as np, torch, pandas as pd
sys.path.insert(0, '/home/server5090/Desktop/M11307320/MATCHA')
from dood.prototype import class_centers

ROOT = '/home/server5090/Desktop/M11307320/MATCHA'
TAG = 'v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix'
RUN = os.path.join(ROOT, 'exp_result_' + TAG)
CSV = os.path.join(ROOT, 'exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_'
                         'aggbn_shadow_osdg_excl_person_seed2026_topo1234_fix/async_diag/shadow_signals.csv')
NCLS, NNODE, NSTY = 6, 9, 3
STY = [0,0,0, 1,1,1, 2,2,2]
SNAME = ['art', 'photo', 'sketch']
FLOOR = float(np.degrees(np.arccos(1 - 1e-7)))

def ang(A, B):
    """逐類別夾角的平均（與 production `_ang` 同式，含同一個 clamp ⇒ 地板 0.0256°）"""
    A = A / np.linalg.norm(A, axis=1, keepdims=True); B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip((A*B).sum(1), -1+1e-7, 1-1e-7))).mean())

def nrm(X): return X / np.linalg.norm(X, axis=-1, keepdims=True)

def load(ep):
    """回傳 own[9][6,128]＝各節點自己的類別中心（階段 1 ⇒ 等於自有畫風欄）"""
    out = []
    for i in range(NNODE):
        suf = 'final' if ep == 'final' else f'epoch_{ep}'
        bs = torch.load(os.path.join(RUN, f'{TAG}_node_{i}_{suf}.pth'),
                        map_location='cpu', weights_only=False)['backbone_state']
        out.append(class_centers(bs['prototypes'].double(), bs['proto_count'].double()).numpy())
    return out

def two_layer(own, owners):
    """兩層平均（plan D6）：先同畫風、再跨畫風。owners ＝ 可採計的節點 index 清單。"""
    per = []
    for s in range(NSTY):
        ks = [k for k in owners if STY[k] == s]
        if ks: per.append(nrm(np.mean([own[k] for k in ks], axis=0)))
    return nrm(np.mean(per, axis=0))

EPS = [50, 100, 150, 200, 'final']
print('=' * 100)
print('§A 3a／3b／3d 預量測 —— 這些訊號是「位準」還是「位移」？')
print('=' * 100)
rows = {k: {} for k in ['3a', '3d', '3b', 'floor']}
for ep in EPS:
    own = load(ep)
    G = two_layer(own, list(range(NNODE)))
    for i in range(NNODE):
        Gmi = two_layer(own, [k for k in range(NNODE) if k != i])
        same = [k for k in range(NNODE) if STY[k] == STY[i] and k != i]
        rows['3a'].setdefault(i, []).append(ang(own[i], G))
        rows['3d'].setdefault(i, []).append(ang(own[i], Gmi))
        rows['3b'].setdefault(i, []).append(ang(own[i], nrm(np.mean([own[k] for k in same], axis=0))))
        allsame = [k for k in range(NNODE) if STY[k] == STY[i]]
        rows['floor'].setdefault(i, []).append(ang(nrm(np.mean([own[k] for k in allsame], axis=0)), G))

for nm in ['3a', '3d', '3b', 'floor']:
    M = np.array([rows[nm][i] for i in range(NNODE)])              # [9, 5]
    print(f'\n── {nm} ──   （列＝節點、欄＝ep50/100/150/200/final，單位 °）')
    for i in range(NNODE):
        print(f'   node{i}({SNAME[STY[i]]:<6}) ' + '  '.join(f'{v:6.2f}' for v in M[i])
              + f'   │ 節點內 全距 {M[i].max()-M[i].min():5.2f}  std {M[i].std(ddof=1):5.2f}')
    bt = M.mean(1)
    print(f'   {"節點間（用各節點時間平均）":<22} 全距 {bt.max()-bt.min():5.2f}  std {bt.std(ddof=1):5.2f}')
    wi = M.std(ddof=1, axis=1).mean()
    print(f'   🚨 **節點間 std {bt.std(ddof=1):.2f}° vs 節點內 std {wi:.2f}°  ⇒ 比值 {bt.std(ddof=1)/max(wi,1e-9):5.2f}×**')

print()
print('=' * 100)
print('§B E6 的門檻：量現行訊號（規則 0／1）的同一個統計量當基準，不要憑感覺定 0.5')
print('=' * 100)
df = pd.read_csv(CSV); d = df[df.is_warm == 0]
dom = d['domain'].values
print('⚠️ E6 現在的分母是 `r3a_floor`，但規則 0／1 沒有「floor」⇒ 跨候選不可比。')
print('   改用**訊號自己的中位數**當分母 ⇒ 無量綱、每個候選都定義得出來、可直接跟現行比。')
print(f'\n{"訊號":<10}{"節點":<9}{"p5":>11}{"p50":>11}{"p95":>11}{"(p95−p5)/p50":>15}')
print('-' * 100)
bench = {}
for nm, col in [('規則0', 'sig_r0'), ('規則1', 'sig_r1')]:
    v = d[col].values.astype(float); rs = []
    for n in sorted(np.unique(dom)):
        m = (dom == n); x = v[m]; x = x[np.isfinite(x)]
        p5, p50, p95 = np.percentile(x, [5, 50, 95]); r = (p95 - p5) / p50
        rs.append(r)
        print(f'{nm:<10}{n:<9}{p5:>11.3e}{p50:>11.3e}{p95:>11.3e}{r:>15.2f}')
    bench[nm] = np.array(rs)
    print(f'{nm+" 全體":<19}{"":>33}中位 {np.median(rs):.2f}   最小 {min(rs):.2f}   最大 {max(rs):.2f}')
print()
print(f'⇒ **現行能運作的訊號，(p95−p5)/p50 落在 {min(bench["規則0"]):.2f}–{max(bench["規則0"]):.2f}（規則0）、'
      f'{min(bench["規則1"]):.2f}–{max(bench["規則1"]):.2f}（規則1）。**')
print(f'⇒ E6 的門檻建議改成：**(p95−p5)/p50 低於「現行規則 0 的逐節點最小值 {min(bench["規則0"]):.2f}」⇒ 淘汰**。')
print('   理由：那是一條**已經在運作**的訊號能撐住的最差動態範圍，不是憑感覺的 0.5。')

print()
print('=' * 100)
print('§C E5 的「明顯弱於」：block bootstrap 的 κ 標準誤')
print('=' * 100)
E = [f'e{i}' for i in range(8)]
nat0 = (d['sig_r0'].values >= 1e-5); fired0 = d['fired'].values.astype(bool)
steps = d['step'].values; ep_ = d['epoch'].values
nr = d['n_received'].values.astype(float); rc = np.zeros(len(d))
for n in sorted(np.unique(dom)):
    m = np.where(dom == n)[0]; a = 0.0
    for i in m:
        a += nr[i]; rc[i] = a
        if fired0[i]: a = 0.0
s1 = d['sig_r1'].values.astype(float)
raw = d[[f'r2c_{e}' for e in E]].to_numpy(dtype=float)
with np.errstate(all='ignore'): s2 = np.nanmean(raw, 1)

def kap(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean()*b.mean() + (1-a.mean())*(1-b.mean())
    return (po - pe)/(1 - pe) if pe < 1 else np.nan
def dec(sig, idx, target):
    x = sig[idx]; return x >= np.nanquantile(x, 1 - target)

# 區塊 ＝ (節點, epoch)：資料在節點內、epoch 內自相關，區塊重抽才不會低估誤差
blocks = {}
for j, (n, e) in enumerate(zip(dom, ep_)):
    blocks.setdefault((n, e), []).append(j)
keys = list(blocks.keys())
rng = np.random.default_rng(20261002)
print(f'區塊數（節點 × epoch）＝ {len(keys)}，每塊約 {len(d)/len(keys):.0f} 列')
targets = {'規則0 vs 累積收到': nat0, '規則1 vs 累積收到': s1, '規則2c vs 累積收到': s2}
B = 300
for nm, sg in targets.items():
    base_idx = np.arange(len(d))
    drc = dec(rc, base_idx, nat0.mean())
    obs = kap(nat0, drc) if nm.startswith('規則0') else kap(dec(sg, base_idx, nat0[base_idx].mean()), drc)
    bs = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([blocks[keys[k]] for k in pick])
        t = nat0[idx].mean()
        dr = dec(rc, idx, t)
        a = nat0[idx] if nm.startswith('規則0') else dec(sg, idx, t)
        k = kap(a, dr)
        if np.isfinite(k): bs.append(k)
    bs = np.array(bs)
    print(f'  {nm:<22} κ = {obs:.4f}   bootstrap SE = {bs.std(ddof=1):.4f}   '
          f'95% CI [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]')
se0 = None
print()
print('⇒ E5 的「明顯弱於規則 0 的 0.2678」建議寫死成：')
print('   **候選的 κ 必須低於 0.2678 − 2·SE(差值)**；而差值的 SE 要對**同一次重抽**算（配對），')
print('   不是把兩個邊際 SE 相加 —— 兩個 κ 共用同一批列、是配對的（看板 📌「共用隨機來源要用配對誤差」）。')
# 配對差值
for nm, sg in [('規則1', s1), ('規則2c', s2)]:
    ds = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([blocks[keys[k]] for k in pick])
        t = nat0[idx].mean(); dr = dec(rc, idx, t)
        k0 = kap(nat0[idx], dr); kx = kap(dec(sg, idx, t), dr)
        if np.isfinite(k0) and np.isfinite(kx): ds.append(k0 - kx)
    ds = np.array(ds)
    print(f'   配對差值 κ(規則0,累積) − κ({nm},累積) ＝ {ds.mean():+.4f} ± {ds.std(ddof=1):.4f}（SE）'
          f'  ⇒ 2·SE ＝ {2*ds.std(ddof=1):.4f}')
print('=' * 100)
