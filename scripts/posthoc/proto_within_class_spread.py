# -*- coding: utf-8 -*-
"""原型為什麼晃：單張圖的特徵離自己類別中心多遠？（2026-10-06，解釋 1006 固定模型測試的 6° 地板）
口徑與訓練時餵原型的特徵相同：訓練增強、train() 模式、BN momentum=0、乾淨前向 → project（單位向量）。
另用「EMA 等效張數」(1+m)/(1−m) 做粗估：晃動 ≈ 單張偏離的垂直分量 / √等效張數（推估，非量測）。"""
import os, sys, glob, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import util
from dood.prototype import init_prototype_buffers
RUN = ('exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_'
       'osdg_excl_person_seed2026_topo1234_fix')
dev, E, NC = 'cuda', 200, 6
args = torch.load(glob.glob(f'{RUN}/*_node_0_epoch_{E}.pth')[0], map_location='cpu', weights_only=False)['args']
dl, n2d = util.load_dataset_single_process(args)
m = args.proto_m; neff = (1 + m) / (1 - m)
print(f'proto_m={m}  EMA 等效張數 (1+m)/(1-m) = {neff:.1f}  半衰期 = {np.log(0.5)/np.log(m):.1f} 張  batch={args.bs}')
for j in range(9):
    sd = torch.load(glob.glob(f'{RUN}/*_node_{j}_epoch_{E}.pth')[0], map_location='cpu', weights_only=False)['backbone_state']
    model = util.select_model(NC, args).to(dev); init_prototype_buffers(model, *sd['prototypes'].shape, device=dev)
    model.load_state_dict(sd, strict=True); model.train()
    for mm in model.modules():
        if isinstance(mm, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)): mm.momentum = 0.0
    torch.manual_seed(5); random.seed(5); np.random.seed(5)
    Z, Y = [], []
    with torch.no_grad():
        for b in dl[f'node_{j}'][0]:
            x, y = util.unpack_batch(b)[:2]
            z3 = model.forward_to_layer3_style(x.to(dev), communicator=None, debug_style_shift=False, iter_num=1, rank=j)
            Z.append(model.project(model.forward_from_layer3(z3)[1]).float()); Y.append(y.to(dev))
    Z, Y = torch.cat(Z), torch.cat(Y)
    angs, perp = [], []
    for c in range(NC):
        zc = Z[Y == c]; mu = F.normalize(zc.mean(0), dim=0)
        cos = (zc @ mu).clamp(-1, 1)
        angs.append(torch.rad2deg(torch.arccos(cos)).mean().item())
        perp.append(torch.sqrt((1 - cos ** 2).mean()).item())       # 垂直分量 RMS
        par = cos.mean().item()
    a, p = np.mean(angs), np.mean(perp)
    pred = np.degrees(np.arctan(p / np.sqrt(neff) / 0.8))            # 粗估：垂直誤差/平行長度（平行≈0.8 假設）
    print(f'node_{j} ({n2d[f"node_{j}"]:<12}) 單張到類別中心平均 {a:5.1f}°   垂直分量RMS {p:.3f}   粗估EMA晃動 ≈ {pred:4.1f}°')
