# -*- coding: utf-8 -*-
"""解法可行性：用「固定探針圖」量類別中心變化，能不能去掉抽樣晃動、又追得上真實變化？（2026-10-06）

問題（1006 兩支測試）：EMA 原型只等於約 39 張圖的平均 ⇒ 換一批圖就晃約 6°；真實結構變化每 50 epoch
     只有 10.4°／6.6°／1.6° ⇒ 觸發用的短時間尺度上被晃動淹沒。
解法構想：每個節點固定一小組自己的訓練圖（每類 k 張）當「探針」，不做隨機增強、用推論模式
     （BN 用 running 統計）⇒ 同一組圖、同一個模型算出來的結果必定相同 ⇒ 抽樣晃動＝0；
     模型一變，探針上的類別中心才會變。廣播時只存探針的類別中心（6×128 個數字）。
本腳本只用現有 checkpoint（ep50/100/150/200），回答三件事：
  ① 晃動地板：同一 checkpoint 算兩次，差幾度（應≈0）
  ② 小探針追不追得上「整份資料」的真實變化：k=2/5/10 各抽 20 組固定子集，比對 27 個（節點×區間）
  ③ 推論模式 vs 訓練模式（原型的口徑）量到的變化是否同方向
⚠️ 只有 50 epoch 間隔的 checkpoint ⇒ 答不了「逐步」行為；那要短跑訓練才知道。
"""
import os, sys, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import util
from dood.prototype import init_prototype_buffers

RUN = ('exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_'
       'osdg_excl_person_seed2026_topo1234_fix')
EPOCHS = [50, 100, 150, 200]
KS, NSUB = [1, 2, 5, 10], 20
dev = 'cuda'
args = torch.load(glob.glob(f'{RUN}/*_node_0_epoch_200.pth')[0], map_location='cpu', weights_only=False)['args']
domain_loaders, node_to_domain = util.load_dataset_single_process(args)
NC = 7 - (1 if getattr(args, 'exclude_class', None) else 0)
TEST_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
ang = lambda A, B: torch.rad2deg(torch.arccos((F.normalize(A, dim=-1) * F.normalize(B, dim=-1)).sum(-1)
                                              .clamp(-1 + 1e-7, 1 - 1e-7)))


def node_feats(j, E, mode):
    """節點 j 整份訓練圖（無增強）在 checkpoint E 的投影特徵 [N,128] 與標籤。mode: 'eval' 或 'train'(BN批統計)"""
    sd = torch.load(glob.glob(f'{RUN}/*_node_{j}_epoch_{E}.pth')[0], map_location='cpu',
                    weights_only=False)['backbone_state']
    model = util.select_model(NC, args).to(dev)
    init_prototype_buffers(model, *sd['prototypes'].shape, device=dev)
    model.load_state_dict(sd, strict=True)
    if mode == 'eval':
        model.eval()
    else:
        model.train()
        for m_ in model.modules():
            if isinstance(m_, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m_.momentum = 0.0
    sub = domain_loaders[f'node_{j}'][0].dataset
    base = sub.dataset
    old_tf = base.transform; base.transform = TEST_TF
    assert getattr(base.dataset, 'transform', None) is None, '底層 ImageFolder 也有 transform ⇒ 會重複套用'
    loader = torch.utils.data.DataLoader(sub, batch_size=64, shuffle=False, num_workers=4)
    Z, Y = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = util.unpack_batch(batch)[:2]
            z3 = model.forward_to_layer3_style(x.to(dev), communicator=None, debug_style_shift=False, iter_num=1, rank=j)
            Z.append(model.project(model.forward_from_layer3(z3)[1]).float()); Y.append(y.to(dev))
    base.transform = old_tf
    return torch.cat(Z), torch.cat(Y)


def centers(Z, Y, idx=None):
    if idx is not None:
        Z, Y = Z[idx], Y[idx]
    S = torch.zeros(NC, Z.shape[1], device=dev).index_add_(0, Y, Z)
    return F.normalize(S, dim=1)


gen = torch.Generator().manual_seed(0)
full, probe, floor, train_vs_eval = [], {k: [] for k in KS}, [], []
for j in range(9):
    feats = {E: node_feats(j, E, 'eval') for E in EPOCHS}
    Z2, _ = node_feats(j, 200, 'eval')
    floor.append(float((Z2 - feats[200][0]).abs().max()))                     # ① 同 checkpoint 兩次
    Y = feats[200][1]
    by_c = [torch.nonzero(Y == c).flatten().cpu() for c in range(NC)]
    subs = {k: [torch.cat([bc[torch.randperm(len(bc), generator=gen)[:k]] for bc in by_c]).to(dev)
                for _ in range(NSUB)] for k in KS}
    for a, b in zip(EPOCHS[:-1], EPOCHS[1:]):
        full.append(float(ang(centers(*feats[a]), centers(*feats[b])).mean()))
        for k in KS:
            probe[k].append([float(ang(centers(*feats[a], s), centers(*feats[b], s)).mean()) for s in subs[k]])
    # ③ 訓練模式口徑（批統計 BN）：只比 150→200 與 50→100 的方向
    tr = {E: node_feats(j, E, 'train') for E in (50, 100, 150, 200)}
    train_vs_eval.append([float(ang(centers(*tr[a]), centers(*tr[b])).mean())
                          for a, b in zip(EPOCHS[:-1], EPOCHS[1:])])
    print(f'node_{j} ({node_to_domain[f"node_{j}"]:<12}) 整份資料(推論模式) 50→100 {full[-3]:5.2f}  100→150 {full[-2]:5.2f}'
          f'  150→200 {full[-1]:5.2f} | 訓練模式 {np.round(train_vs_eval[-1], 2).tolist()} | 兩次重算最大差 {floor[-1]:.1e}',
          flush=True)

full = np.array(full)
print('\n' + '=' * 92)
print(f'① 晃動地板：同一 checkpoint、同一組圖重算兩次，特徵最大差 {max(floor):.1e}（≈0 ⇒ 探針法沒有抽樣晃動）')
print('=' * 92)
print('② 小探針 vs 整份資料（27 個「節點×區間」；探針每組固定、跨 checkpoint 用同一批圖）')
print(f'   整份資料的變化：50→100 平均 {full[0::3].mean():.2f}°、100→150 {full[1::3].mean():.2f}°、150→200 {full[2::3].mean():.2f}°')
for k in KS:
    P = np.array(probe[k])                       # [27, NSUB]
    ratio = P / full[:, None]
    rho = [spearmanr(P[:, s], full).correlation for s in range(NSUB)]
    late = ratio[2::3]
    print(f'   每類 {k:>2} 張（共 {6*k:>2} 張）：探針/整份 比值 中位 {np.median(ratio):.2f}（10–90% {np.percentile(ratio,10):.2f}–{np.percentile(ratio,90):.2f}）'
          f'；後段 150→200 中位 {np.median(late):.2f}（10–90% {np.percentile(late,10):.2f}–{np.percentile(late,90):.2f}）'
          f'；排序一致度(Spearman) 中位 {np.median(rho):.2f}')
print('③ 訓練模式（原型口徑）vs 推論模式：九節點平均 ' +
      ' '.join(f'{lab} {np.mean([r[i] for r in train_vs_eval]):.2f}° vs {full[i::3].mean():.2f}°'
               for i, lab in enumerate(['50→100', '100→150', '150→200'])))
print('④ 去掉時間趨勢：同一區間內 9 節點的排序一致度（Spearman，20 組探針的中位）——全部 27 個的相關會被「越早變越多」灌水')
for k in KS:
    P = np.array(probe[k])
    w = [np.median([spearmanr(P[i::3, s], full[i::3]).correlation for s in range(NSUB)]) for i in range(3)]
    print(f'   每類 {k:>2} 張：50→100 {w[0]:+.2f}  100→150 {w[1]:+.2f}  150→200 {w[2]:+.2f}')
