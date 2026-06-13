"""Multi-target generality probe: is the cartoon-only mechanism story universal?

Reuses eval_ksd_compat helpers (rule 4) but sweeps TARGET over all 4 PACS domains,
each on its OWN det-V1 baseline ckpt (the 'control' arm; KSD-on variants exist only
for cartoon). No retraining — pure post-hoc on existing exp_result_v1_stage1_leave_*.

Question: does "witness reliable at target (cos>0, frac_inward=1, |s_tgt|>=|s_clean|)"
+ target distance ordering hold across art/photo/sketch, or is it cartoon-specific?
Watch SKETCH (extreme pure-shape) vs ART (style-ish, near photo) — they bracket cartoon.

Per target T, per source node N (the other 3 domains):
  clean = N test feats ; target = T test feats ; q_N = N's own frozen diffusion
  bw = median-heuristic(clean) ; at t in {10,25,50}:
    [score_dir] cos_to_clean / frac_inward / |s_tgt|/|s_clean|
    [ksd]       ratio_target = KSD_target/KSD_clean
  probe_target = target intrinsic linear separability (per-domain difficulty)

Run: venv_matcha/bin/python eval_ksd_multitarget.py
"""
import os
import numpy as np
import torch

import util
from eval_ksd_compat import (
    DATASET_ROOT, DEVICE, T_LIST, KSD_N, SEED,
    _build_args, extract_features, make_score_fn, score_dir_to_clean,
    bootstrap_ksd, linear_probe_acc,
)
from test_domain_ood_scores import load_checkpoint, load_pacs_test_data
from dood.utils.diffusion import get_diffusion_model
from dood.ksd import median_heruistic

PACS_DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    args = _build_args()
    rows = []

    for target in PACS_DOMAINS:
        ckpt_dir = f"exp_result_v1_stage1_leave_{target}"
        desc = f"v1_stage1_leave_{target}"
        sources = [d for d in PACS_DOMAINS if d != target]
        print(f"\n############ TARGET={target}  (control V1, sources={sources}) ############", flush=True)

        backbone = util.select_model(args.num_classes, args).to(DEVICE)
        diffusion_model = get_diffusion_model(ft_size=512, denoiser_type="unet0d",
                                              diffusion_denoiser_channels=512,
                                              num_diffusion_steps=1000).to(DEVICE)
        for node in sources:
            ckpt_path = os.path.abspath(os.path.join(ckpt_dir, f"{desc}_{node}_final.pth"))
            if not os.path.exists(ckpt_path):
                print(f"[WARN] missing ckpt: {ckpt_path}"); continue
            load_checkpoint(ckpt_path, backbone, diffusion_model, DEVICE)
            diffusion_model.eval()

            clean_f, _     = extract_features(backbone, load_pacs_test_data(DATASET_ROOT, node, 128, 4)[0], 1024)
            tgt_f, tgt_y   = extract_features(backbone, load_pacs_test_data(DATASET_ROOT, target, 128, 4)[0], 1024)

            bw = median_heruistic(clean_f[:KSD_N].to(DEVICE).float(), clean_f[:KSD_N].to(DEVICE).float())
            probe = linear_probe_acc(tgt_f, tgt_y)
            for t in T_LIST:
                sfn = make_score_fn(diffusion_model, t)
                cdir, finw, snt, snc = score_dir_to_clean(diffusion_model, sfn, tgt_f, clean_f)
                gen = torch.Generator().manual_seed(SEED)
                kc, _ = bootstrap_ksd(clean_f, sfn, bw, gen)
                kt, _ = bootstrap_ksd(tgt_f, sfn, bw, gen)
                rt = kt / (kc + 1e-12)
                print(f"[score_dir] T={target} src={node} t={t} cos_to_clean={cdir:+.4f} "
                      f"frac_inward={finw:.3f} |s_tgt|={snt:.4f} |s_clean|={snc:.4f} "
                      f"(|s|ratio={snt/(snc+1e-8):.3f}) | ratio_target={rt:.3f} probe={probe:.3f}", flush=True)
                rows.append(dict(target=target, node=node, t=t, cos_to_clean=cdir, frac_inward=finw,
                                 s_tgt=snt, s_clean=snc, ratio_target=rt, probe_target=probe))

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/ksd_multitarget.csv", index=False)
    print("\n================ SUMMARY (avg over 3 source nodes) ================")
    print(f"{'target':12s} {'t':>3s}  {'cos_to_clean':>12s} {'frac_inward':>11s} {'|s|ratio':>9s} "
          f"{'ratio_target':>12s} {'probe':>6s}")
    for target in PACS_DOMAINS:
        for t in T_LIST:
            sub = df[(df.target == target) & (df.t == t)]
            if len(sub) == 0:
                continue
            sr = (sub.s_tgt / (sub.s_clean + 1e-8)).mean()
            print(f"{target:12s} {t:>3d}  {sub.cos_to_clean.mean():>+12.4f} {sub.frac_inward.mean():>11.3f} "
                  f"{sr:>9.3f} {sub.ratio_target.mean():>12.3f} {sub.probe_target.mean():>6.3f}")
    print("\nSaved: results/ksd_multitarget.csv")


if __name__ == "__main__":
    main()
