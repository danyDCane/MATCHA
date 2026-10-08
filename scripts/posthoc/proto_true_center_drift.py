# -*- coding: utf-8 -*-
"""去掉抽樣晃動後，類別中心在訓練中後段實際移動多少？（2026-10-06，接 proto_ema_sampling_jitter_frozen.py）

1006 固定模型測試：訓練後段原型觸發量 100% 可由抽樣晃動重現（比值 1.01–1.04）⇒ 真正的結構變化被淹沒。
本腳本改用「節點整份訓練資料的類別平均」當中心（不用 EMA），同一組圖、同一順序、同一組隨機增強
（每個 checkpoint 前都重設同一個 seed ⇒ 配對），只換 checkpoint ⇒ 角度差只來自模型改變。
另用不同 seed 在 E=200 再算一次 ⇒ 這個量自己的地板（增強＋批次組成造成）。
特徵口徑與訓練時原型相同：train() 模式、BN momentum=0、乾淨前向 → project。
"""
import os, sys, glob, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import util
from dood.prototype import init_prototype_buffers

RUN = ('exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_'
       'osdg_excl_person_seed2026_topo1234_fix')
EPOCHS = [50, 100, 150, 200]
dev = 'cuda'
ck0 = torch.load(glob.glob(f'{RUN}/*_node_0_epoch_200.pth')[0], map_location='cpu', weights_only=False)
args = ck0['args']
domain_loaders, node_to_domain = util.load_dataset_single_process(args)
NC = 7 - (1 if getattr(args, 'exclude_class', None) else 0)
ang = lambda A, B: torch.rad2deg(torch.arccos((F.normalize(A, dim=-1) * F.normalize(B, dim=-1)).sum(-1)
                                              .clamp(-1 + 1e-7, 1 - 1e-7)))


def centers(j, E, seed):
    sd = torch.load(glob.glob(f'{RUN}/*_node_{j}_epoch_{E}.pth')[0], map_location='cpu',
                    weights_only=False)['backbone_state']
    model = util.select_model(NC, args).to(dev)
    init_prototype_buffers(model, *sd['prototypes'].shape, device=dev)
    model.load_state_dict(sd, strict=True)
    model.train()
    for m_ in model.modules():
        if isinstance(m_, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m_.momentum = 0.0
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    S = torch.zeros(NC, sd['prototypes'].shape[-1], device=dev); n = torch.zeros(NC, device=dev)
    with torch.no_grad():
        for batch in domain_loaders[f'node_{j}'][0]:
            x, y = util.unpack_batch(batch)[:2]; x, y = x.to(dev), y.to(dev)
            z3 = model.forward_to_layer3_style(x, communicator=None, debug_style_shift=False, iter_num=1, rank=j)
            z = model.project(model.forward_from_layer3(z3)[1])
            S.index_add_(0, y, z); n += torch.bincount(y, minlength=NC).float()
    return F.normalize(S / n[:, None], dim=1), n


rows = []
print(f'{"節點":<8}{"畫風":<13}' + ''.join(f'{f"{a}→{b}":>11}' for a, b in zip(EPOCHS[:-1], EPOCHS[1:]))
      + f'{"地板(換seed)":>14}{"每類張數":>16}')
for j in range(9):
    C = {E: centers(j, E, 7)[0] for E in EPOCHS}
    Cb, n = centers(j, 200, 8)
    d = [float(ang(C[a], C[b]).mean()) for a, b in zip(EPOCHS[:-1], EPOCHS[1:])]
    fl = float(ang(C[200], Cb).mean())
    rows.append(d + [fl])
    print(f'node_{j:<3}{node_to_domain[f"node_{j}"]:<13}' + ''.join(f'{v:11.2f}' for v in d) + f'{fl:14.2f}'
          + f'{int(n.min()):>8}–{int(n.max())}', flush=True)
R = np.array(rows)
print('-' * 80)
print('九節點平均' + ' ' * 12 + ''.join(f'{v:11.2f}' for v in R.mean(0)))
print('\n對照：EMA 原型觸發量的抽樣晃動地板（1006 固定模型測試）後段約 6°（photo 4.8／art 5.9／sketch 6.5）')
print('讀法：「150→200」是 1,000 步（50 epoch × 20 步）累積的真實結構變化；若它和地板同量級或更小，')
print('      則在觸發會用到的時間尺度（幾步到幾十步）上，真實變化遠小於 EMA 晃動。')
