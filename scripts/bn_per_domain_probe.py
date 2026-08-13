"""逐域 BN 探針 —— 回答「是否只有 sketch sync/async 有差、為何唯獨 sketch 爛」。

對 sync/async × seed2026/seed42、每個節點：
  orig    = 原生 BN eval cartoon（＝acc.log 末 epoch）
  srcBN   = 凍權重、BN 重估於「該節點自己的源域」eval cartoon（＝測試現實的地板）
  tgtBN   = 凍權重、BN 重估於 cartoon（天花板/oracle）
  d_src2cart = ‖source-BN − cartoon-BN‖（該域 source 風格離 cartoon 多遠）
按域群(art/photo/sketch)平均。

判：
  art/photo orig: sync≈async? → 確認差異只在 sketch
  d_src2cart: sketch 最大? → 「離 cartoon 最遠 = 手銬最重」
  地板(srcBN) 的 sync-vs-async gap: 只在 sketch 出現? → async 耐受只在大錯配時有用
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from bn_signal_gating import build_backbone, load_backbone_only, eval_acc, reconstruct_node_to_domain
from bn_recalib_probe import recompute_bn, capture_bn, node_files
from test_domain_ood_scores import load_pacs_test_data

SETTINGS = [
    ("sync-s2026",  "exp_result_v1_stage2_leave_cartoon_seed2026_topo1234"),
    ("async-s2026", "exp_result_v1_stage2_leave_cartoon_async_const_tau1e-5_style_seed2026_topo1234"),
    ("sync-s42",    "exp_result_v1_stage2_leave_cartoon_seed42_topo1234"),
    ("async-s42",   "exp_result_v1_stage2_leave_cartoon_async_const_tau1e-5_style_seed42_topo1234"),
]
GROUPS = {"art": [0, 1, 2], "photo": [3, 4, 5], "sketch": [6, 7, 8]}
DR = "../datasets/"


def main():
    dev = "cuda"
    node_to_domain, _ = reconstruct_node_to_domain("cartoon", 9)
    loaders = {d: load_pacs_test_data(DR, d, 64, 4)[0]
               for d in ["art_painting", "photo", "sketch", "cartoon"]}

    table = {}  # (setting, group) -> dict of means
    for label, ckpt_dir in SETTINGS:
        files = node_files(ckpt_dir, 9)
        ck0 = torch.load(files[0], map_location="cpu", weights_only=False)
        nc = ck0["backbone_state"]["backbone.fc.weight"].shape[0]
        bb = build_backbone(ck0["args"], nc, dev)
        per = {}
        for j in range(9):
            dom = node_to_domain[f"node_{j}"]
            load_backbone_only(files[j], bb, dev); orig = eval_acc(bb, loaders["cartoon"])
            load_backbone_only(files[j], bb, dev); recompute_bn(bb, loaders[dom], dev)
            src_bn = capture_bn(bb); srcacc = eval_acc(bb, loaders["cartoon"])
            load_backbone_only(files[j], bb, dev); recompute_bn(bb, loaders["cartoon"], dev)
            tgt_bn = capture_bn(bb); tgtacc = eval_acc(bb, loaders["cartoon"])
            per[j] = (orig, srcacc, tgtacc, float(np.linalg.norm(src_bn - tgt_bn)))
        for g, nodes in GROUPS.items():
            arr = np.array([per[j] for j in nodes]).mean(0)
            table[(label, g)] = dict(orig=arr[0], src=arr[1], tgt=arr[2], d=arr[3])
        print(f"[{label}] done")

    print("\n================ 逐域摘要 ================")
    print(f"{'group':7s} {'setting':12s} {'orig':>6s} {'地板srcBN':>9s} {'天花板tgt':>9s} {'d(src→cart)':>11s}")
    for g in GROUPS:
        for label, _ in SETTINGS:
            t = table[(label, g)]
            print(f"{g:7s} {label:12s} {t['orig']:6.1f} {t['src']:9.1f} {t['tgt']:9.1f} {t['d']:11.2f}")
        print()

    print("================ 判決 ================")
    for seed in ["s2026", "s42"]:
        print(f"\n--- seed {seed} ---")
        for g in GROUPS:
            sy = table[(f"sync-{seed}", g)]; asy = table[(f"async-{seed}", g)]
            print(f"  {g:7s}: orig Δ(async−sync)={asy['orig']-sy['orig']:+5.1f}  "
                  f"地板Δ={asy['src']-sy['src']:+5.1f}  天花板Δ={asy['tgt']-sy['tgt']:+5.1f}  "
                  f"d(src→cart) sync={sy['d']:.1f}")
    print("\n預期驗證：art/photo orig Δ≈0（sync≈async）、sketch Δ 明顯；"
          "d(src→cart) sketch>>art（sketch 離 cartoon 最遠＝手銬最重）；地板 Δ 只在 sketch 大。")


if __name__ == "__main__":
    main()
