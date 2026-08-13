"""BN 「更貼 source 還是更貼 target」探針 —— 解 dany 的反駁。

dany 反駁：少融合(async) → BN 更貼本地 source → 對未知風格(cartoon) 應更差。
但數據相反(async sketch 60 > sync 54)。隱藏假設＝「BN 貼 source 風格」。
訓練時 style_shift 開著 → BN 累積的是「被廣泛風格擾動的 source 活化」、非原始 source。

量每個 sketch 節點的『訓練後 BN(orig)』離：
  source-BN = 凍權重、BN 重估於 sketch(該節點來源域、乾淨無 aug)
  target-BN = 凍權重、BN 重估於 cartoon(target)
的 L2 距離；並測 source-BN 配到 cartoon 的 acc。

判：
  d_target(async) < d_target(sync)  → async 訓練後 BN 較貼 target ⇒ 我對、dany 的假設被否
  d_source(async) < d_source(sync)  → async 較貼 source ⇒ dany 對(但需再解釋為何仍泛化好)

sanity：target-BN acc on cartoon 應重現 bn_recalib_probe 的 oracle 值(sync~82.5/async~83.3)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from bn_signal_gating import build_backbone, load_backbone_only, eval_acc
from bn_recalib_probe import recompute_bn, capture_bn, node_files
from test_domain_ood_scores import load_pacs_test_data

SETTINGS = [
    ("sync",       "exp_result_v1_stage2_leave_cartoon_seed2026_topo1234"),
    ("model_only", "exp_result_v1_stage2_leave_cartoon_async_const_tau1e-5_seed2026_topo1234_modelonly"),
    ("full_async", "exp_result_v1_stage2_leave_cartoon_async_const_tau1e-5_style_seed2026_topo1234"),
]
SKETCH = [6, 7, 8]
DATAROOT = "../datasets/"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cartoon_loader, _ = load_pacs_test_data(DATAROOT, "cartoon", 64, 4)   # target
    sketch_loader, _ = load_pacs_test_data(DATAROOT, "sketch", 64, 4)     # source（乾淨 sketch）

    agg = {}
    for label, ckpt_dir in SETTINGS:
        files = node_files(ckpt_dir, 9)
        ck0 = torch.load(files[0], map_location="cpu", weights_only=False)
        num_class = ck0["backbone_state"]["backbone.fc.weight"].shape[0]
        backbone = build_backbone(ck0["args"], num_class, device)

        print(f"\n===== {label} =====")
        print(f"{'node':8s} {'d_source':>9s} {'d_target':>9s} {'closer':>8s} "
              f"{'orig_acc':>9s} {'srcBN_acc':>9s} {'tgtBN_acc':>9s}")
        rows = []
        for j in SKETCH:
            # trained (orig) BN
            load_backbone_only(files[j], backbone, device)
            trained = capture_bn(backbone)
            orig_acc = eval_acc(backbone, cartoon_loader)

            # source-BN（重估於 sketch）
            load_backbone_only(files[j], backbone, device)
            recompute_bn(backbone, sketch_loader, device)
            source_bn = capture_bn(backbone)
            src_acc = eval_acc(backbone, cartoon_loader)

            # target-BN（重估於 cartoon）
            load_backbone_only(files[j], backbone, device)
            recompute_bn(backbone, cartoon_loader, device)
            target_bn = capture_bn(backbone)
            tgt_acc = eval_acc(backbone, cartoon_loader)

            d_src = float(np.linalg.norm(trained - source_bn))
            d_tgt = float(np.linalg.norm(trained - target_bn))
            closer = "source" if d_src < d_tgt else "target"
            print(f"node_{j:<2d} {d_src:9.3f} {d_tgt:9.3f} {closer:>8s} "
                  f"{orig_acc:9.2f} {src_acc:9.2f} {tgt_acc:9.2f}")
            rows.append((d_src, d_tgt, orig_acc, src_acc, tgt_acc))
        arr = np.array(rows)
        agg[label] = arr.mean(0)
        print(f"  sketch群均: d_source={agg[label][0]:.3f} d_target={agg[label][1]:.3f} "
              f"orig={agg[label][2]:.2f} srcBN={agg[label][3]:.2f} tgtBN={agg[label][4]:.2f}")

    print("\n================ 判決 (sketch 群均) ================")
    print(f"{'':12s} {'d_source':>9s} {'d_target':>9s} {'orig':>7s} {'srcBN':>7s} {'tgtBN':>7s}")
    for label, _ in SETTINGS:
        a = agg[label]
        print(f"{label:12s} {a[0]:9.3f} {a[1]:9.3f} {a[2]:7.2f} {a[3]:7.2f} {a[4]:7.2f}")
    ds_sync, dt_sync = agg["sync"][0], agg["sync"][1]
    ds_asy, dt_asy = agg["full_async"][0], agg["full_async"][1]
    print(f"\n  dany 預測: async 較貼 source ⇒ d_source(async) < d_source(sync)?"
          f"  {ds_asy:.2f} vs {ds_sync:.2f} → {'成立(dany對)' if ds_asy<ds_sync else '否(dany此假設錯)'}")
    print(f"  我的說法: async 較貼 target ⇒ d_target(async) < d_target(sync)?"
          f"  {dt_asy:.2f} vs {dt_sync:.2f} → {'成立(我對)' if dt_asy<dt_sync else '否'}")
    print(f"\n  [sanity] tgtBN acc 應≈ oracle(sync~82.5 / async~83.3): "
          f"sync={agg['sync'][4]:.2f} async={agg['full_async'][4]:.2f}")


if __name__ == "__main__":
    main()
