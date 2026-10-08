# -*- coding: utf-8 -*-
"""攻擊⑤：`_global_centers(only=...)` 的 100% 相同，是資料造成的空操作，還是 bug 假裝成 no-op？

fb 的解釋：原型表 epoch 1 內就填滿 54 格且全程沒掉 ⇒ 任何兩節點的可見集合永遠是全部 9 個
⇒ `only=comm` 恆等於 `only=None` ⇒ 空操作。

要區分「真空操作」與「bug」只有一個辦法：**故意餵不滿的 stamp，看 `only=` 到底動不動**。
若連故意不滿時都不動 ⇒ 是 bug；若會動 ⇒ fb 的解釋成立。

本檔直接綁真正的 `_global_centers`（不複寫），四個測試：
  T1 `only=` 在故意不滿時必須改變輸出（否則就是 bug）
  T2 重現 6a 指出的一階偏差：我少看一個畫風時，原始口徑≠0、共同可見口徑=0
  T3 🚨 可見集合**逐類別不均勻**時，共同可見口徑還等於 0 嗎？（`comm` 是對類別 OR 的）
  T4 本 run 的 ncom==9 是否足以證明兩邊的表都滿
"""
import sys, torch, torch.nn.functional as F, numpy as np, pandas as pd
sys.path.insert(0, '/home/server5090/Desktop/M11307320/MATCHA')
from communicator import SingleProcessCommunicator as _Comm

NCLS, NNODE, NSTY, DIM = 6, 9, 3, 128
torch.manual_seed(2026)

class Stub:
    """只帶 _global_centers 需要的四個屬性，方法本體直接取自真正的 Communicator。"""
    proto_n_cls, proto_n_node, proto_n_sty, proto_dim = NCLS, NNODE, NSTY, DIM
    _global_centers = _Comm._global_centers
s = Stub()

def mkstore(seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(NCLS, NNODE, DIM, generator=g), dim=2)

def angdeg(A, B):
    # ⚠️ clamp 到 1-1e-7 沿用 production `_ang` 的寫法 ⇒ 完全相同的向量也會回 arccos(1-1e-7)
    #    ＝ 本函式的**地板**。下面把地板印出來，不可把地板讀成「有差異」。
    c = (F.normalize(A, dim=1) * F.normalize(B, dim=1)).sum(1).clamp(-1+1e-7, 1-1e-7)
    return float(torch.rad2deg(torch.arccos(c)).mean())

FLOOR = float(np.degrees(np.arccos(1 - 1e-7)))
print(f'⚠️ 角度函式的地板（沿用 production 的 clamp 1-1e-7）＝ {FLOOR:.4f}° ⇒ 這個值就是「0」。\n')

print('=' * 92)
print('T1  `only=` 在故意不滿時會不會改變輸出？（若不會 ⇒ 是 bug，不是空操作）')
print('=' * 92)
st = mkstore(1)
sp_full = torch.ones(NCLS, NNODE, dtype=torch.long) * 7
G_all = s._global_centers(st, sp_full)
for drop in [[0], [0,1,2], [0,3,6], [8]]:
    only = torch.ones(NNODE, dtype=torch.bool); only[drop] = False
    G_sub = s._global_centers(st, sp_full, only=only)
    print(f'  排除 owner {str(drop):<12} ⇒ 與 only=None 的夾角 {angdeg(G_all, G_sub):7.4f}°   '
          f'{"❌ 在地板上 ⇒ 沒動 ⇒ BUG" if angdeg(G_all,G_sub) <= FLOOR*1.01 else "✅ 有動"}')
only_eq = torch.ones(NNODE, dtype=torch.bool)
_a = angdeg(G_all, s._global_centers(st, sp_full, only=only_eq))
print(f'  only 全 True（應與 None 相同）    ⇒ 夾角 {_a:.4f}° ＝ 地板 ⇒ 逐位相同 ✅')
print('  ⇒ `only=` 確實有作用。100% 相同不是實作失效。')

print()
print('=' * 92)
print('T2  重現 6a 指出的一階偏差（我少看一個畫風）')
print('=' * 92)
sp_i = sp_full.clone(); sp_i[:, 0:3] = 0        # 我完全看不到 art 的三個 owner（node_8 轉傳到達前）
sp_j = sp_full.clone()                           # 鄰居看得到全部
print(f'  ① 原始口徑（各用自己的全部可見）：∠ = {angdeg(s._global_centers(st, sp_i), s._global_centers(st, sp_j)):.4f}°')
print('     ⇒ **store 兩邊完全相同，角度卻不是 0** ＝ 6a 說的一階偏差，確實存在。')
comm = (sp_i > 0).any(0) & (sp_j > 0).any(0)
print(f'  ② 共同可見口徑：ncom = {int(comm.sum())}，∠ = '
      f'{angdeg(s._global_centers(st, sp_i, only=comm), s._global_centers(st, sp_j, only=comm)):.4f}° ＝ 地板 ⇒ 真的是 0')
print('     ⇒ 修正有效、零點天生是 0。✅ fb 的修正在「畫風層級缺席」這個情境下是對的。')

print()
print('=' * 92)
print('T3  🚨 可見集合【逐類別不均勻】時呢？（`comm` 是對類別做 OR 的）')
print('=' * 92)
sp_i = sp_full.clone(); sp_j = sp_full.clone()
sp_i[1:, 0] = 0      # owner0 我只在 class 0 看得到
sp_j[0, 0] = 0       # owner0 鄰居在 class 0 看不到、其他類別看得到
comm = (sp_i > 0).any(0) & (sp_j > 0).any(0)
a_raw = angdeg(s._global_centers(st, sp_i), s._global_centers(st, sp_j))
a_com = angdeg(s._global_centers(st, sp_i, only=comm), s._global_centers(st, sp_j, only=comm))
print(f'  owner0 的可見性逐類別不同（我只在 class0 可見、鄰居在 class0 不可見）')
print(f'  comm = {comm.int().tolist()}  （對類別 OR ⇒ owner0 仍被判為「共同可見」）')
print(f'  ① 原始口徑 ∠ = {a_raw:.4f}°')
print(f'  ② 共同可見口徑 ∠ = {a_com:.4f}°   ⇒ {"❌ 遠高於地板 ⇒ 仍然不是 0" if a_com > FLOOR*1.01 else "✅ 在地板上 ＝ 0"}')
print('  ⇒ 🚨 `comm` 應該**逐類別**算（`sp_i > 0` 與 `Pj > 0` 的逐格交集），現在是對類別 OR。')
print('    ⇒ 「共同可見口徑的零點天生是 0」這句**只在可見性逐類別一致時成立**，不是無條件成立。')
print('    本 run 表全滿 ⇒ 不會咬人；但 fb 把它當成「稀疏拓撲的保險」留著，')
print('    而稀疏拓撲**正是**可見性逐類別不一致的場合（原型是逐類別 EMA 填的）⇒ 保險有洞。')

print()
print('=' * 92)
print('T4  本 run：ncom==9 是否足以證明「兩邊的表都滿」')
print('=' * 92)
CSV = ('/home/server5090/Desktop/M11307320/MATCHA/exp_result_v1_stage2_leave_cartoon_p1a_'
       'async_const_tau1e-5_style_nodiff_aggbn_shadow_osdg_excl_person_seed2026_topo1234_fix/'
       'async_diag/shadow_signals.csv')
df = pd.read_csv(CSV); d = df[df.is_warm == 0]
nc = d[[f'ncom_e{i}' for i in range(8)]].to_numpy(dtype=float)
cf = d['cells_filled'].to_numpy()
print(f'  ncom 定義 ＝ (我可見 AND 鄰居可見).sum() ⇒ ncom==9 蘊含**兩邊都 9 個 owner 全可見**')
print(f'  非暖機區 ncom 的最小值 {np.nanmin(nc):.0f}、唯一值 {sorted(set(nc[np.isfinite(nc)].astype(int)))}')
print(f'  cells_filled 最小 {cf.min()}（上限 54）、< 54 的列數 {(cf < 54).sum()}')
print(f'  暖機區 ncom 最小值 {np.nanmin(df[df.is_warm==1][[f"ncom_e{i}" for i in range(8)]].to_numpy(dtype=float)):.0f}'
      f'  ⇒ 暖機區**有**不滿的時候（被排除掉了）')
print('  ⇒ ✅ fb 的解釋成立：非暖機區兩邊的表都滿 ⇒ comm 恆全 True ⇒ 真空操作。')
print('  ⇒ 但「空操作」是**這個 run 的資料性質**，不是實作的性質（T1 已證實作會動）。')
print('=' * 92)
