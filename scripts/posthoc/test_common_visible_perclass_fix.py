# -*- coding: utf-8 -*-
"""驗證 2026-10-02 的逐類別共同可見修正（matcha-02 的 T3 洞）

T3 構造：owner 0 只有 class 0 我看得到、鄰居看不到，其餘類別兩邊都看得到。
  舊實作：comm = vis_i & vis_j（**對類別 OR**）⇒ owner 0 被判共同可見
          ⇒ 我的 G 在 class 0 含 owner 0、鄰居的不含 ⇒ 共同可見口徑 ≡ 原始口徑 ⇒ 完全沒修到
  新實作：comm 逐格 [C, n_node] ⇒ class 0 的 owner 0 被排除 ⇒ 兩邊同集合 ⇒ 回到地板
⚠️ 角度函式沿用 production 的 clamp(1-1e-7) ⇒ 完全相同的向量也回 ~0.026°，以下「地板」即指此。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, torch.nn.functional as F
from communicator import SingleProcessCommunicator

C, N, P = 6, 9, 128
torch.manual_seed(2026)

class Shim(SingleProcessCommunicator):
    def __init__(self):
        self.proto_n_cls, self.proto_n_node, self.proto_n_sty, self.proto_dim = C, N, 3, P

sh = Shim()
ang = lambda A, B: float(torch.rad2deg(torch.arccos(
    ((F.normalize(A, dim=1) * F.normalize(B, dim=1)).sum(1)).clamp(-1+1e-7, 1-1e-7))).mean())

store = F.normalize(torch.randn(C, N, P), dim=2)
sp = torch.ones(C, N, dtype=torch.long)          # 我：全部看得到
Pj = torch.ones(C, N, dtype=torch.long)          # 鄰居：全部看得到
Pj[0, 0] = 0                                     # 只有 (class 0, owner 0) 鄰居看不到

G_i, G_j = sh._global_centers(store, sp), sh._global_centers(store, Pj)
print(f'原始口徑（各用自己看得到的全部）            {ang(G_i, G_j):8.4f}°')

old_comm = (sp > 0).any(0) & (Pj > 0).any(0)                       # 舊：對類別 OR
old = ang(sh._global_centers(store, sp, only=old_comm.unsqueeze(0).expand(C, N)),
          sh._global_centers(store, Pj, only=old_comm.unsqueeze(0).expand(C, N)))
print(f'舊實作 共同可見（節點級遮罩）                {old:8.4f}°   ⇒ 與原始口徑相同 ＝ 沒修到')

new_comm = (sp > 0) & (Pj > 0)                                     # 新：逐格 AND
new = ang(sh._global_centers(store, sp, only=new_comm),
          sh._global_centers(store, Pj, only=new_comm))
print(f'新實作 共同可見（逐類別遮罩）                {new:8.4f}°   ⇒ 回到地板 ＝ 修好')
print(f'ncom（六類全部共同可見的 owner 數）          {int(new_comm.all(0).sum())}  （粒度錯配時會從 9 掉下來）')

# 對照：粒度一致（整個 owner 缺席）時新舊必須相同
Pk = torch.ones(C, N, dtype=torch.long); Pk[:, 0] = 0
o_old = (sp > 0).any(0) & (Pk > 0).any(0)
a_old = ang(sh._global_centers(store, sp, only=o_old.unsqueeze(0).expand(C, N)),
            sh._global_centers(store, Pk, only=o_old.unsqueeze(0).expand(C, N)))
o_new = (sp > 0) & (Pk > 0)
a_new = ang(sh._global_centers(store, sp, only=o_new), sh._global_centers(store, Pk, only=o_new))
print(f'\n對照（整個 owner 0 缺席、粒度一致）  原始 {ang(G_i, sh._global_centers(store, Pk)):7.4f}°'
      f'  舊 {a_old:7.4f}°  新 {a_new:7.4f}°  ⇒ 新舊一致、都修好 ✅')
