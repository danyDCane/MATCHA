# -*- coding: utf-8 -*-
"""階段一b 的事前零點檢查：3a/3b/3c/3d 在「零延遲」下還剩多少角度？（📌F3：算得出來的別跑）

`matcha-6a` 2026-10-02 的預測：
  3a `∠(C_i, G_i)` 會把 11.03° 的跨畫風地板帶回來 —— 從一個頂點到三個兩兩相距 11° 的點的質心
  ≈ 11.03/√3 ≈ 6.4°，**與學習無關、與延遲無關**；而且畫風越非典型地板越大
  ⇒ 訊號會編碼「你是哪個畫風」，跟規則 2c 編碼「你的度數」是同一個病。

做法：拿九個節點 **epoch_200 的自有原型欄**（全部當下、零延遲、零 staleness），
      直接算四個候選訊號。**剩下的就是地板。**
  地板 ≈ 0  ⇒ 該候選的零點是 0、可以實作
  地板 >> 0 且逐畫風系統性不同 ⇒ **事前判掉、不必實作**
"""
import os, sys, glob, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, torch.nn.functional as F
import numpy as np

RUN = ('exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_'
       'osdg_excl_person_seed2026_topo1234_fix')
NC, NS, PER = 6, 3, 3
STY = ['art', 'photo', 'sketch']

def own_col(sd, j):
    """節點 j 的自有原型欄：[6, P]，L2 正規化；proto_count==0 的格回 None"""
    bs = sd['backbone_state']
    p, c = bs['prototypes'], bs['proto_count']
    dom = j // PER                                   # 節點 j 的來源畫風索引
    assert (c[:, dom] > 0).all(), f'node {j} 有空格'
    return F.normalize(p[:, dom, :].float(), dim=1)  # [6, P]

C = []
for j in range(9):
    f = glob.glob(os.path.join(RUN, f'*_node_{j}_epoch_200.pth'))
    assert len(f) == 1, f
    sd = torch.load(f[0], map_location='cpu', weights_only=False)
    assert sd['domain'] == f'node_{j}', sd['domain']
    C.append(own_col(sd, j))
C = torch.stack(C)                                   # [9, 6, P]
print(f'載入九個節點 epoch_200 的自有原型欄：{tuple(C.shape)}（全部當下、零延遲）\n')

ang = lambda A, B: torch.rad2deg(torch.arccos(
    (F.normalize(A, dim=-1) * F.normalize(B, dim=-1)).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)))

def two_level(idx):
    """兩層平均（先同畫風、再跨畫風），只採計 idx 裡的節點"""
    ms = []
    for s in range(NS):
        ks = [k for k in idx if k // PER == s]
        if ks: ms.append(F.normalize(C[ks].mean(0), dim=1))
    return F.normalize(torch.stack(ms).mean(0), dim=1)   # [6, P]

print('=' * 86)
print('零延遲地板（九個節點、六類別平均；單位：度）')
print('=' * 86)
print(f'{"節點":<8}{"畫風":<9}{"3a 全體":>10}{"3d 排除自己":>13}{"3b 同畫風(-i)":>15}{"3c 自身漂移":>13}')
res = {k: [] for k in ('3a', '3d', '3b')}
for j in range(9):
    a  = float(ang(C[j], two_level(list(range(9)))).mean())
    dd = float(ang(C[j], two_level([k for k in range(9) if k != j])).mean())
    same = [k for k in range(9) if k // PER == j // PER and k != j]
    b  = float(ang(C[j], F.normalize(C[same].mean(0), dim=1)).mean())
    res['3a'].append(a); res['3d'].append(dd); res['3b'].append(b)
    print(f'node_{j:<3}{STY[j//PER]:<9}{a:>10.3f}{dd:>13.3f}{b:>15.3f}{0.0:>13.3f}')
for k in ('3a', '3d', '3b'):
    v = np.array(res[k]); print(f'{k} 全體平均 {v.mean():.3f}°（範圍 {v.min():.3f}–{v.max():.3f}）')
print('3c 零延遲地板 ＝ 0（定義上：G_i(t) 與 G_i(t_last) 是同一個量在兩個時點）')

print('\n' + '=' * 86)
print('逐畫風：地板會不會編碼「你是哪個畫風」？')
print('=' * 86)
print(f'{"畫風":<9}{"3a":>9}{"3d":>9}{"3b":>9}')
for s in range(NS):
    ks = [k for k in range(9) if k // PER == s]
    print(f'{STY[s]:<9}' + ''.join(f'{np.mean([res[k][i] for i in ks]):>9.3f}' for k in ('3a','3d','3b')))
for k in ('3a', '3d', '3b'):
    pm = [np.mean([res[k][i] for i in range(9) if i//PER == s]) for s in range(NS)]
    v = np.array(res[k])
    bet = np.var(pm, ddof=1); wit = np.mean([np.var([res[k][i] for i in range(9) if i//PER==s], ddof=1) for s in range(NS)])
    print(f'  {k}: 畫風間極差 {max(pm)-min(pm):.3f}°｜組間/組內變異比 F ＝ {bet/wit:.2f}'
          f'  {"🚨 地板隨畫風系統性不同" if bet/wit > 4 else "✅ 畫風不主導"}')

print('\n' + '=' * 86)
print('對照：matcha-6a 的解析預測與既有量測')
print('=' * 86)
s_gap = 11.03
print(f'  既有實測 純畫風間距 {s_gap}°、純節點雜訊 σ 5.12°（0930_proto_same_vs_cross_style.log）')
print(f'  解析預測 3a ≈ 11.03/√3 ＝ {s_gap/np.sqrt(3):.2f}°   實測 {np.mean(res["3a"]):.3f}°')
print(f'  解析預測 3b ≈ 5.12/√2 ＝ {5.12/np.sqrt(2):.2f}°（只剩節點雜訊）   實測 {np.mean(res["3b"]):.3f}°')

# ============================================================================
# 追加（2026-10-02，依 matcha-02 的 E7 型別檢查 ＋ 弦長平方扣底 ＋ 本檔補的 3c）
# ============================================================================
print('\n' + '=' * 86)
print('追加一：E7 型別檢查（matcha-02）—— 訊號分出來的是「節點身分」還是「時間變化」？')
print('=' * 86)
EPS = [50, 100, 150, 200]
Cs = {}
for e in EPS:
    L = []
    for j in range(9):
        f = glob.glob(os.path.join(RUN, f'*_node_{j}_epoch_{e}.pth'))
        sd = torch.load(f[0], map_location='cpu', weights_only=False)
        L.append(own_col(sd, j))
    Cs[e] = torch.stack(L)

def sig(name, Cm, j):
    if name == '3a':  return float(ang(Cm[j], tl(Cm, list(range(9)))).mean())
    if name == '3d':  return float(ang(Cm[j], tl(Cm, [k for k in range(9) if k != j])).mean())
    if name == '3b':
        sm = [k for k in range(9) if k // PER == j // PER and k != j]
        return float(ang(Cm[j], F.normalize(Cm[sm].mean(0), dim=1)).mean())
def tl(Cm, idx):
    ms = []
    for s in range(NS):
        ks = [k for k in idx if k // PER == s]
        if ks: ms.append(F.normalize(Cm[ks].mean(0), dim=1))
    return F.normalize(torch.stack(ms).mean(0), dim=1)

print(f'{"訊號":<8}{"節點間 std":>12}{"節點內 std":>12}{"比值":>8}   判讀（≥1 ⇒ 編碼節點身分）')
for nm in ('3a', '3d', '3b'):
    M = np.array([[sig(nm, Cs[e], j) for e in EPS] for j in range(9)])   # [9, 4]
    sb, sw = M.mean(1).std(ddof=1), M.std(1, ddof=1).mean()
    print(f'{nm:<8}{sb:>12.3f}{sw:>12.3f}{sb/sw:>8.2f}   '
          f'{"🚨 節點身分主宰" if sb/sw >= 1 else "✅ 時間變化主宰"}')

print('\n' + '=' * 86)
print('追加二：弦長平方扣掉畫風底後，「我相對全體」還剩多少（角度不可相減）')
print('=' * 86)
ch2 = lambda deg: (2 * np.sin(np.radians(deg) / 2)) ** 2
inv = lambda c: np.degrees(2 * np.arcsin(np.sqrt(max(c, 0)) / 2))
print(f'{"畫風":<9}{"3a":>8}{"畫風底":>9}{"殘餘(正交)":>12}{"σ_node":>9}{"殘餘/σ_node":>13}')
for s in range(NS):
    ks = [k for k in range(9) if k // PER == s]
    a = np.mean([res['3a'][k] for k in ks])
    # 畫風底 ＝ 我這一組的中心（同畫風三節點平均）相對全體 G 的角度
    fl = float(ang(F.normalize(C[ks].mean(0), dim=1), two_level(list(range(9)))).mean())
    r = inv(ch2(a) - ch2(fl))
    print(f'{STY[s]:<9}{a:>8.2f}{fl:>9.2f}{r:>12.2f}{5.12:>9.2f}{r/5.12:>13.2f}')
print('🚨 殘餘 ＝ 扣掉畫風後真正屬於「我個人」的部分；若 < σ_node ⇒ 可用訊號比節點雜訊還小。')

print('\n' + '=' * 86)
print('追加三：3c 的量級（兩人都沒量）—— 全域估計在 epoch 尺度上動多少')
print('=' * 86)
print('⚠️ 這是【上界】：3c 的實際作用尺度是廣播間隔（中位 4 步），比 50 epoch 短得多。')
print(f'{"區間":<16}{"3c ＝ ∠(G,G)":>14}{"規則1 等價 ∠(C_i,C_i) 平均":>26}{"比值":>8}')
for a, b in zip(EPS[:-1], EPS[1:]):
    g = float(ang(tl(Cs[b], list(range(9))), tl(Cs[a], list(range(9)))).mean())
    c = float(np.mean([float(ang(Cs[b][j], Cs[a][j]).mean()) for j in range(9)]))
    print(f'ep{a:>3}→{b:<9}{g:>14.3f}{c:>26.3f}{g/c:>8.2f}')
print('⇒ 比值 ＝ 「大家整體走的距離」/「我自己走的距離」。遠小於 1 ⇒ 個別節點的漂移在平均中互相抵消')
print('   ⇒ 3c 的動態範圍會比規則 1 小，但它的零點是 0（沒有常數底）⇒ 兩件事要分開判。')
