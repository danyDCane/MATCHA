# -*- coding: utf-8 -*-
"""原型觸發量的「6° 地板」是不是抽樣晃動？—— 固定模型重算原型（2026-10-06）

背景：`proto_trigger_signal_floor.py`（1005）量到訓練後段 sig_r1 在距上次廣播 4–6 步就停在約 6°、
     隔 50 步也一樣 ⇒ 懷疑原型 EMA（proto_m=0.95、逐樣本）只等於約 39 張圖的平均，換一批圖就晃。

做法：載入 `_fix` run 各節點 epoch E 的 checkpoint，**模型參數完全不動**，只照訓練時的乾淨前向
     （train() 模式、BN momentum 暫設 0、forward_to_layer3_style(communicator=None) → forward_from_layer3
      → project）＋ ema_prototypes_live（m=0.95）往下走 T 步，記每步的類別中心（class_centers）。
     兩種圖片順序（seed 0／1）各走一次。
     ⇒ 模型不變，所以量到的任何角度都只來自「抽到哪些圖＋隨機增強」。

輸出：
  ① 同一條序列內、間隔 g 步的夾角（六類平均，度）——與訓練中 shadow_signals 的同一個量逐格對照
  ② 兩條獨立序列同一步的夾角（純抽樣地板）
判準（事前寫死）：後段（E=150/200）間隔 ≥4 步的各欄，固定模型／訓練實測
  ≥ 0.85 ⇒ 後段訊號幾乎全是抽樣晃動；≤ 0.5 ⇒ 晃動只占小部分、平台另有原因；介於中間 ⇒ 混合
"""
import os, sys, glob, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import util
from dood.prototype import class_centers, init_prototype_buffers, ema_prototypes_live

RUN = ('exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_'
       'osdg_excl_person_seed2026_topo1234_fix')
EPOCHS = [50, 150, 200]
T, BURN, SEEDS = 60, 5, [0, 1]
BUCKETS = [(1, 1, '1'), (2, 2, '2'), (3, 3, '3'), (4, 6, '4-6'), (7, 10, '7-10'), (11, 20, '11-20'), (21, 50, '21-50')]
# 訓練實測（1005 log，ep151-200 / ep1-50，中位數，度）
TRAIN_LATE = {'1': 3.99, '2': 5.06, '3': 5.58, '4-6': 5.92, '7-10': 6.00, '11-20': 5.91, '21-50': 5.86}
TRAIN_EARLY = {'1': 5.05, '2': 6.80, '3': 7.76, '4-6': 8.29}
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def ang(A, B):
    c = (F.normalize(A, dim=-1) * F.normalize(B, dim=-1)).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.rad2deg(torch.arccos(c)).mean(-1)          # 六類平均


ck0 = torch.load(glob.glob(f'{RUN}/*_node_0_epoch_200.pth')[0], map_location='cpu', weights_only=False)
args = ck0['args']
loaded = util.load_dataset_single_process(args)
domain_loaders, node_to_domain = loaded if isinstance(loaded, tuple) else (loaded, {k: k for k in loaded})
proto_domain_list = sorted(set(node_to_domain.values()))
num_classes = 7 - (1 if getattr(args, 'exclude_class', None) else 0)
print(f'node_to_domain={node_to_domain}  proto_domain_list={proto_domain_list}  num_classes={num_classes}')

res = {}   # (E, node) -> dict
for E in EPOCHS:
    for j in range(9):
        node = f'node_{j}'
        ck = torch.load(glob.glob(f'{RUN}/*_node_{j}_epoch_{E}.pth')[0], map_location='cpu', weights_only=False)
        sd = ck['backbone_state']
        model = util.select_model(num_classes, args).to(dev)
        _c, _d, _p = sd['prototypes'].shape
        init_prototype_buffers(model, _c, _d, _p, device=dev)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        crit = [k for k in missing if k.startswith(('proj_head', 'prototypes', 'proto_count'))]
        assert not crit, crit
        if E == EPOCHS[0] and j == 0:
            print(f'load: missing={len(missing)} unexpected={len(unexpected)} {missing[:3]} {unexpected[:3]}')
        model.train()
        for m_ in model.modules():
            if isinstance(m_, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m_.momentum = 0.0
        p0 = {n: p.detach().clone() for n, p in model.named_parameters()}
        dom_idx = proto_domain_list.index(node_to_domain[node])
        loader = domain_loaders[node][0]
        traj = {}
        for s in SEEDS:
            torch.manual_seed(1000 + s); random.seed(1000 + s); np.random.seed(1000 + s)
            protos, counts = model.prototypes.detach().clone(), model.proto_count.detach().clone()
            C = [class_centers(protos, counts)]
            it = iter(loader)
            with torch.no_grad():
                for t in range(T):
                    try:
                        batch = next(it)
                    except StopIteration:
                        it = iter(loader); batch = next(it)
                    data, target = util.unpack_batch(batch)[:2]
                    data, target = data.to(dev), target.to(dev)
                    z3 = model.forward_to_layer3_style(data, communicator=None, debug_style_shift=False,
                                                       iter_num=t + 1, rank=j)
                    _, vec = model.forward_from_layer3(z3)
                    z = model.project(vec)
                    protos, counts = ema_prototypes_live(protos, counts, z, target, dom_idx, args.proto_m)
                    C.append(class_centers(protos, counts))
            traj[s] = torch.stack(C)                         # [T+1, 6, P]
        # 模型參數真的沒動（BN running stats 也不動）
        drift = max(float((p.detach() - p0[n]).abs().max()) for n, p in model.named_parameters())
        assert drift == 0.0, drift
        out = {}
        for lo, hi, lab in BUCKETS:
            v = []
            for s in SEEDS:
                X = traj[s]
                for g in range(lo, hi + 1):
                    for t in range(BURN, T + 1 - g):
                        v.append(float(ang(X[t], X[t + g])))
            out[lab] = np.median(v) if v else np.nan
        out['cross'] = float(np.median([float(ang(traj[0][t], traj[1][t])) for t in range(BURN, T + 1)]))
        res[(E, node)] = out
        print(f'E={E:3d} {node} ({node_to_domain[node]:<12}) ' +
              ' '.join(f'{lab}:{out[lab]:5.2f}' for _, _, lab in BUCKETS) + f'  兩序列同步:{out["cross"]:5.2f}', flush=True)
        del model; torch.cuda.empty_cache()

print('\n' + '=' * 96)
print('① 固定模型：同一序列內間隔 g 步的夾角（九節點中位數的中位數，度）vs 訓練實測')
print('=' * 96)
for E in EPOCHS:
    ref = TRAIN_EARLY if E == 50 else TRAIN_LATE
    tag = '訓練 ep1-50' if E == 50 else '訓練 ep151-200'
    row = {lab: np.median([res[(E, f'node_{j}')][lab] for j in range(9)]) for _, _, lab in BUCKETS}
    print(f'E={E:3d} 固定模型   ' + ' '.join(f'{lab}:{row[lab]:5.2f}' for _, _, lab in BUCKETS))
    print(f'      {tag:<11}' + ' '.join(f'{lab}:{ref[lab]:5.2f}' if lab in ref else f'{lab}:  —  ' for _, _, lab in BUCKETS))
    print('      比值(固定/訓練) ' + ' '.join(f'{lab}:{row[lab]/ref[lab]:5.2f}' if lab in ref else f'{lab}:  —  ' for _, _, lab in BUCKETS))
    cr = np.median([res[(E, f'node_{j}')]['cross'] for j in range(9)])
    print(f'      兩條獨立序列同一步的夾角（純抽樣地板）：{cr:.2f}°')

print('\n' + '=' * 96)
print('② 逐畫風（E=150/200 平均，間隔 21-50 步）—— 訓練實測後段：art 5.7–6.2°、photo 4.8–5.1°、sketch 6.4–6.8°')
print('=' * 96)
for sty in proto_domain_list:
    nodes = [f'node_{j}' for j in range(9) if node_to_domain[f'node_{j}'] == sty]
    v = [res[(E, n)]['21-50'] for E in (150, 200) for n in nodes]
    print(f'  {sty:<13} {np.round(v, 2).tolist()}  平均 {np.mean(v):.2f}°')

print('\n判準（事前寫死）：後段間隔 ≥4 步各欄 比值 ≥0.85 ⇒ 幾乎全是抽樣晃動；≤0.5 ⇒ 晃動只占小部分；中間 ⇒ 混合')
