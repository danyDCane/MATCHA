"""換卡末位差的機制檢查（1028，零重訓）：同一批圖、同一個模型，只改批次大小，各層輸出是否逐位相同。

背景：5090→5080 換卡後，暖機前權重逐位相同、只有原型紀錄差 1–2 位（1026 報告 §4）；baseline 沒有原型、
ep1–16 逐字相同（1027）。NVIDIA cuBLAS 文件：同一版工具在「同架構、同 SM 數」的卡上才保證逐位相同
（5090 170 個 SM、5080 84 個）。矩陣乘法會依「問題形狀＋SM 數」挑不同的切分方式 ⇒ 加總順序不同 ⇒ 末位不同。
本檢查在 5080 上單機示範：哪些層的輸出會隨「計算方式的選擇」改變（這裡用批次大小觸發）。

用法：venv_matcha/bin/python scripts/posthoc/proj_head_shape_sensitivity_check.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
os.chdir(ROOT)

import torch
import util
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion

torch.backends.cudnn.deterministic = True     # 與 train.py 相同
torch.backends.cudnn.benchmark = False
N = "v1_stage2_leave_sketch_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234"
m, _ = load_backbone_diffusion(f"exp_result_{N}/{N}_node_0_epoch_200.pth", 6, "cuda")
m.eval()
x, _, _ = util.unpack_batch(next(iter(TD.load_pacs_test_data("../datasets/", "art_painting", 64, 2)[0])))
x = x.cuda()
print(f"GPU={torch.cuda.get_device_name(0)} SM={torch.cuda.get_device_properties(0).multi_processor_count}")


@torch.no_grad()
def run(xx):
    logits, vec = m.forward_from_layer3(m.forward_to_layer3_style(xx, communicator=None))
    h = m.proj_head[0](vec)
    h2 = m.proj_head[2](torch.relu(h))
    return vec, logits, h, h2


full = run(x)
for b in (1, 7, 23, 33, 48):
    part = run(x[:b])
    same = [torch.equal(p, f[:b]) for p, f in zip(part, full)]
    md = max(float((p - f[:b]).abs().max()) for p, f in zip(part[2:], full[2:]))
    print(f"批次 {b:2d} vs 64：骨幹特徵={same[0]} 分類頭(512→6)={same[1]} "
          f"投影層第一層(512→512)={same[2]} 第二層(512→128)={same[3]} 投影層最大差={md:.2e}")
