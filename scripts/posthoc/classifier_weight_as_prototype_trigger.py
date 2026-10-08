# -*- coding: utf-8 -*-
"""不晃、零額外前向的「類別原型」：分類層權重能不能代替 EMA 原型當觸發量？（2026-10-06）

動機：EMA 原型是「從樣本估出來」的 ⇒ 必晃（1006 固定模型測試）；探針前向不晃但要多花約 3,150 倍計算。
     分類層權重 backbone.fc.weight（6×512）是參數 ⇒ 精確、不晃、不用前向、只占 12 KB。
     神經崩塌理論（訓練後期分類層權重會對齊各類別平均特徵的方向）⇒ 它本身就是「權重空間的類別原型」。
本腳本用現有 checkpoint（ep50/100/150/200，9 節點）回答：
  ① 對齊：fc 第 c 列 和「第 c 類 512 維平均特徵（扣掉全體平均）」的夾角 —— 對得越齊，越能說它是原型
  ② 追蹤：27 個（節點×區間）上，fc 權重的變化 vs 真實類別中心的變化（整份資料、同圖、推論模式）
     vs 整個模型參數的變化 —— 排序一致度（Spearman），分「全部」與「同一區間內 9 節點」兩種
⚠️ 只有 50 epoch 間隔 ⇒ 看不到逐步行為；但 fc 是參數，「不晃」由構造保證，不需逐步驗。
"""
import os, sys, glob
import numpy as np, torch, torch.nn.functional as F
from torchvision import transforms
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import util
from dood.prototype import init_prototype_buffers

RUN = ('exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_'
       'osdg_excl_person_seed2026_topo1234_fix')
EPOCHS = [50, 100, 150, 200]
dev, NC = 'cuda', 6
args = torch.load(glob.glob(f'{RUN}/*_node_0_epoch_200.pth')[0], map_location='cpu', weights_only=False)['args']
dl, n2d = util.load_dataset_single_process(args)
TEST_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
ang = lambda A, B: torch.rad2deg(torch.arccos((F.normalize(A, dim=-1) * F.normalize(B, dim=-1)).sum(-1)
                                              .clamp(-1 + 1e-7, 1 - 1e-7)))


def load(j, E):
    sd = torch.load(glob.glob(f'{RUN}/*_node_{j}_epoch_{E}.pth')[0], map_location='cpu',
                    weights_only=False)['backbone_state']
    model = util.select_model(NC, args).to(dev)
    init_prototype_buffers(model, *sd['prototypes'].shape, device=dev)
    model.load_state_dict(sd, strict=True); model.eval()
    return model, sd


def feats(model, j):
    sub = dl[f'node_{j}'][0].dataset; base = sub.dataset
    old = base.transform; base.transform = TEST_TF
    V, Z, Y = [], [], []
    with torch.no_grad():
        for b in torch.utils.data.DataLoader(sub, batch_size=64, shuffle=False, num_workers=4):
            x, y = util.unpack_batch(b)[:2]
            z3 = model.forward_to_layer3_style(x.to(dev), communicator=None, debug_style_shift=False, iter_num=1, rank=j)
            _, v = model.forward_from_layer3(z3)
            V.append(v.float()); Z.append(model.project(v).float()); Y.append(y.to(dev))
    base.transform = old
    return torch.cat(V), torch.cat(Z), torch.cat(Y)


def cmeans(X, Y):
    return torch.stack([X[Y == c].mean(0) for c in range(NC)])


rows, align = [], {E: [] for E in EPOCHS}
for j in range(9):
    st = {}
    for E in EPOCHS:
        model, sd = load(j, E)
        V, Z, Y = feats(model, j)
        mu = cmeans(V, Y); muc = mu - mu.mean(0, keepdim=True)
        W = sd['backbone.fc.weight'].to(dev).float()
        align[E].append(float(ang(W, muc).mean()))
        flat = torch.cat([v.flatten().float() for k, v in sd.items()
                          if v.dtype.is_floating_point and not k.startswith(('prototypes', 'proto_count'))
                          and not k.endswith(('running_mean', 'running_var'))])
        st[E] = dict(W=W, mu=muc, C=F.normalize(cmeans(Z, Y), dim=1), flat=flat.to(dev))
        del model
    for a, b in zip(EPOCHS[:-1], EPOCHS[1:]):
        rows.append(dict(node=j, iv=f'{a}→{b}',
                         fc=float(ang(st[a]['W'], st[b]['W']).mean()),
                         mu512=float(ang(st[a]['mu'], st[b]['mu']).mean()),
                         c128=float(ang(st[a]['C'], st[b]['C']).mean()),
                         param=float((st[b]['flat'] - st[a]['flat']).norm() / st[a]['flat'].numel() ** 0.5)))
    print(f'node_{j} ({n2d[f"node_{j}"]:<12}) ' + '  '.join(
        f'{r["iv"]}: fc {r["fc"]:5.2f}° 中心512 {r["mu512"]:5.2f}° 中心128 {r["c128"]:5.2f}°' for r in rows[-3:]), flush=True)

print('\n' + '=' * 96)
print('① 分類層權重 vs 類別平均特徵（扣全體平均）的夾角（九節點平均；90°＝無關，越小越對齊）')
print('=' * 96)
print('   ' + '   '.join(f'ep{E}: {np.mean(align[E]):5.1f}°（{min(align[E]):.1f}–{max(align[E]):.1f}）' for E in EPOCHS))
print('\n' + '=' * 96)
print('② 27 個（節點×區間）的變化量：排序一致度 Spearman')
print('=' * 96)
k = ['fc', 'mu512', 'c128', 'param']
lab = {'fc': '分類層權重', 'mu512': '512維類別中心', 'c128': '128維類別中心(原型空間)', 'param': '整個模型參數'}
for iv in sorted({r['iv'] for r in rows}, key=lambda s: int(s.split('→')[0])):
    R = [r for r in rows if r['iv'] == iv]
    print(f'   {iv}: ' + '  '.join(f'{lab[x]} 平均 {np.mean([r[x] for r in R]):.4g}' for x in k))
for x, y in [('fc', 'c128'), ('fc', 'mu512'), ('param', 'c128'), ('fc', 'param')]:
    allr = spearmanr([r[x] for r in rows], [r[y] for r in rows]).correlation
    within = [spearmanr([r[x] for r in rows if r['iv'] == iv], [r[y] for r in rows if r['iv'] == iv]).correlation
              for iv in ['50→100', '100→150', '150→200']]
    print(f'   {lab[x]} vs {lab[y]}：全部 {allr:+.2f}｜同區間內 9 節點 ' + ' / '.join(f'{w:+.2f}' for w in within))
