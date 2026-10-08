# -*- coding: utf-8 -*-
"""觸發檢查一次要花多少？參數位移 vs 探針前向 vs 參數草圖（2026-10-06，dany 問「多跑一次前向值得嗎」）

量四種「判斷要不要送」的做法，單次成本（GPU 牆鐘中位數、計算量、要多存多少）：
  A 現行 `_param_delta` 原樣（逐張量 .item()，communicator.py）
  A' 同一個量的向量化寫法（判準⑥：稱職實作者會怎麼寫）
  P 探針前向：30 張圖、推論模式、不算梯度，再算類別中心夾角
  S 參數草圖：把「目前參數」即時雜湊成 k=768 個桶（count sketch），與上次送出時存的 768 個數比
  ＋對照：一步訓練的近似成本（兩次前向＋一次反向，batch 64）
模型＝`_fix` run node_0 epoch 200（ResNet-18＋投影頭，結構與訓練相同）；輸入用隨機張量（時間與內容無關）。
"""
import os, sys, glob, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import util
from dood.prototype import init_prototype_buffers

RUN = ('exp_result_v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_shadow_'
       'osdg_excl_person_seed2026_topo1234_fix')
dev = 'cuda'
ck = torch.load(glob.glob(f'{RUN}/*_node_0_epoch_200.pth')[0], map_location='cpu', weights_only=False)
args, sd = ck['args'], ck['backbone_state']
NC = 6
model = util.select_model(NC, args).to(dev)
init_prototype_buffers(model, *sd['prototypes'].shape, device=dev)
model.load_state_dict(sd, strict=True)
params = [p for n, p in model.named_parameters() if not n.startswith('diffusion_model.')]
names = [n for n, p in model.named_parameters() if not n.startswith('diffusion_model.')]
n_param = sum(p.numel() for p in params)
ref = {n: p.detach().clone() + 1e-4 * torch.randn_like(p) for n, p in zip(names, params)}   # 假裝訓練過一段
ref_list = [ref[n] for n in names]
K = 768


def bench(fn, reps=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


def delta_original():                       # communicator._param_delta 原樣
    sq, n = 0.0, 0
    for name, p in model.named_parameters():
        if name.startswith('diffusion_model.'):
            continue
        d = p.data - ref[name]
        sq += float((d * d).sum().item()); n += d.numel()
    return (sq ** 0.5) / (n ** 0.5)


def delta_vectorized():
    d = torch._foreach_sub([p.data for p in params], ref_list)
    return float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(d)))) / n_param ** 0.5


_ar = {i: torch.arange(p.numel(), device=dev, dtype=torch.int64) for i, p in enumerate(params)}   # 只為公平計時；部署時可分塊即時生成


def sketch_of(tensors):
    s = torch.zeros(K, device=dev)
    for i, p in enumerate(tensors):
        h = (_ar[i] * 2654435761 + 97 * i + 13) % 2147483647
        sign = ((h >> 16) & 1).float() * 2 - 1
        s.index_add_(0, h % K, sign * p.reshape(-1))
    return s


ref_sketch = sketch_of(ref_list)


def delta_sketch():
    return float(torch.linalg.vector_norm(sketch_of([p.data for p in params]) - ref_sketch)) / n_param ** 0.5


x30 = torch.randn(30, 3, 224, 224, device=dev)
y30 = torch.arange(30, device=dev) % NC
ref_centers = F.normalize(torch.randn(NC, 128, device=dev), dim=1)


def probe_forward():
    model.eval()
    with torch.no_grad():
        z3 = model.forward_to_layer3_style(x30, communicator=None, debug_style_shift=False, iter_num=1, rank=0)
        z = model.project(model.forward_from_layer3(z3)[1])
        C = F.normalize(torch.zeros(NC, z.shape[1], device=dev).index_add_(0, y30, z), dim=1)
        return float(torch.rad2deg(torch.arccos((C * ref_centers).sum(1).clamp(-1, 1))).mean())


x64 = torch.randn(64, 3, 224, 224, device=dev)
y64 = torch.arange(64, device=dev) % NC
opt = torch.optim.SGD(params, lr=0.0)       # lr=0：只量時間、不改模型


def train_step_proxy():
    model.train()
    opt.zero_grad(set_to_none=True)
    loss = 0
    for _ in range(2):                       # 乾淨前向＋擾動前向（近似；不含風格模組與原型損失）
        z3 = model.forward_to_layer3_style(x64, communicator=None, debug_style_shift=False, iter_num=1, rank=0)
        logits, vec = model.forward_from_layer3(z3)
        loss = loss + F.cross_entropy(logits, y64) + 0 * model.project(vec).sum()
    loss.backward(); opt.step()


# 計算量（FlopCounterMode 只數矩陣乘／卷積；位移與草圖是逐元素，用解析式）
with FlopCounterMode(display=False) as fc:
    probe_forward()
probe_flops = fc.get_total_flops()
with FlopCounterMode(display=False) as fc:
    train_step_proxy()
step_flops = fc.get_total_flops()
delta_flops = 3 * n_param                    # 減、平方、加總
sketch_flops = 6 * n_param                   # 雜湊（乘、加、取餘×2）＋乘號＋累加，粗估

t = {
    'A 參數位移（現行原樣）': bench(delta_original),
    "A' 參數位移（向量化）": bench(delta_vectorized),
    'S 參數草圖 k=768': bench(delta_sketch),
    'P 探針前向 30 張': bench(probe_forward),
    '一步訓練（近似）': bench(train_step_proxy, reps=30),
}
mem = {
    'A 參數位移（現行原樣）': 4 * n_param, "A' 參數位移（向量化）": 4 * n_param,
    'S 參數草圖 k=768': 4 * K, 'P 探針前向 30 張': 4 * NC * 128, '一步訓練（近似）': None,
}
fl = {
    'A 參數位移（現行原樣）': delta_flops, "A' 參數位移（向量化）": delta_flops,
    'S 參數草圖 k=768': sketch_flops, 'P 探針前向 30 張': probe_flops, '一步訓練（近似）': step_flops,
}
print(f'參數量 n = {n_param:,}（不含 diffusion；與 _param_delta 範圍相同）')
print(f'{"做法":<22}{"牆鐘(ms)":>10}{"占一步訓練":>12}{"計算量(GFLOP)":>16}{"要多存(bytes)":>18}')
for k_ in t:
    share = f'{100 * t[k_] / t["一步訓練（近似）"]:.1f}%'
    m_ = f'{mem[k_]:,}' if mem[k_] is not None else '—'
    print(f'{k_:<22}{t[k_]:>10.2f}{share:>12}{fl[k_] / 1e9:>16.3f}{m_:>18}')
print(f'\n探針前向 / 參數位移 的計算量比 ＝ {probe_flops / delta_flops:,.0f} 倍')
print('⚠️ 草圖的 arange 索引為了計時預先建好（約 4×n bytes 暫存）；部署要分塊即時生成，時間會再多一些。')
print('⚠️ 一步訓練是近似（無風格模組、無原型損失、無 diffusion），真實訓練步更貴 ⇒ 「占一步訓練」偏高估。')
