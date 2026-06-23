"""
Open-Set DG (route-A) evaluation for MATCHA, post-hoc on an OSDG-trained checkpoint.

Protocol:
  - Metric provenance: OSCR (primary) follows the centralized OSDG-PACS standard
    MEDIC (ICCV'23) / EBiL-HaDS (NeurIPS'24); OSCR itself is Dhamija (NeurIPS'18).
    The 6-known + person-unknown LOO setup is shared with federated OSDG
    (ICME-AABAW'25) -- but AABAW reports H-score/UNK/ALL acc, NOT OSCR.
  - Model trained on source domains with ONE class held out (--exclude_class person, 6-way).
  - Test = unseen (leave_out) domain's FULL test set:
        known  = labels != unknown_idx  -> must be classified correctly AND accepted
        unknown= labels == unknown_idx  -> must be REJECTED (person, never trained)
  - Reject score g(x): diffusion eps_mse (HIGHER=OOD) vs baselines MSP / energy.
  - Metrics: OSCR (threshold-free, primary), H-score (best-tau, aux), closed-set known acc (R0),
             AUROC(known vs unknown) per score (R3 pure-detection sanity).
  - per source-node model, then node-mean (centralized graphid=-1 first run).

Reuses joint_eval_mixed_stream.score_and_predict / oriented_conf and CA/TD loaders. No training.
"""
import os
import sys
import csv
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import util
import collapse_analysis as CA
import test_domain_ood_scores as TD
from joint_eval_mixed_stream import score_and_predict

PACS = CA.PACS


def load_backbone_diffusion(ckpt_path, num_classes, device):
    backbone = util.select_model(num_classes, CA.build_backbone_args()).to(device)
    diffusion = CA.get_diffusion_model(
        ft_size=512, denoiser_type="unet0d",
        diffusion_denoiser_channels=512, num_diffusion_steps=1000,
    ).to(device)
    CA.load_checkpoint(ckpt_path, backbone, diffusion, device)
    backbone.eval(); diffusion.eval()
    return backbone, diffusion


def compute_oscr(reject_score, pred, label, unknown_idx):
    """Threshold-free OSCR (Dhamija NeurIPS'18). reject_score: HIGHER = more OOD/reject.
    x-axis = FPR (unknown accepted), y-axis = CCR (known correct AND accepted). Returns area."""
    known = label != unknown_idx
    n_k = max(int(known.sum()), 1)
    n_u = max(int((~known).sum()), 1)
    correct = (pred == label) & known
    conf = -reject_score                          # higher conf = accept first
    order = np.argsort(-conf, kind="stable")
    correct_o = correct[order].astype(float)
    known_o = known[order]
    ccr = np.concatenate([[0.0], np.cumsum(correct_o) / n_k])
    fpr = np.concatenate([[0.0], np.cumsum((~known_o).astype(float)) / n_u])
    return float(np.trapz(ccr, fpr))


def compute_hscore_best(reject_score, pred, label, unknown_idx):
    """H = 2*accK*accU/(accK+accU), accK=known correct&accepted, accU=unknown rejected.
    Swept over thresholds (accept if reject_score<=tau); returns best H + that (accK,accU)."""
    known = label != unknown_idx
    n_k = max(int(known.sum()), 1)
    n_u = max(int((~known).sum()), 1)
    correct = (pred == label) & known
    thr = np.unique(reject_score)
    best_h, best = 0.0, (0.0, 0.0)
    for t in thr:
        accept = reject_score <= t
        accK = float((correct & accept).sum()) / n_k
        accU = float(((~known) & (~accept)).sum()) / n_u
        if accK + accU > 0:
            h = 2 * accK * accU / (accK + accU)
            if h > best_h:
                best_h, best = h, (accK, accU)
    return best_h, best[0], best[1]


def compute_fpr_at_tpr(reject_score, is_unknown, tpr_target=0.95):
    """FPR (known accepted as OOD) at the threshold giving TPR>=tpr_target on unknowns.
    reject_score: HIGHER = more OOD. is_unknown: bool array (positive=OOD). Lower=better."""
    pos = reject_score[is_unknown]                 # unknowns
    neg = reject_score[~is_unknown]                # knowns
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(pos, 1.0 - tpr_target)       # accept >=tpr_target of unknowns as OOD
    return float((neg >= thr).mean())              # knowns wrongly flagged as OOD


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leave_out", required=True, choices=PACS)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--datasetRoot", default="../datasets/")
    p.add_argument("--num_classes", type=int, default=6, help="classifier head width (6 for OSDG person-held-out)")
    p.add_argument("--unknown_idx", type=int, default=6, help="ImageFolder label of held-out unknown (person=6)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_eval_steps", type=int, default=25)
    p.add_argument("--ood_eval_scores_type", default="eps_mse")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--num_nodes", type=int, default=0,
                   help="0=centralized (nodes=source domains, ckpt {desc}_{domain}); "
                        ">0=P2P virtual-node (nodes=node_0..node_{N-1}, ckpt {desc}_node_{i})")
    p.add_argument("--ckpt_epoch", default="final",
                   help="'final' or an epoch number N -> loads _epoch_{N}.pth (for DG->detection "
                        "transfer curve across the every-50 checkpoints)")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    diff_steps = list(range(args.num_eval_steps))
    root = args.datasetRoot

    # Node set + checkpoint key. Centralized: one model per source domain. P2P: node_0..node_{N-1}.
    if args.num_nodes and args.num_nodes > 0:
        node_keys = [f"node_{i}" for i in range(args.num_nodes)]
        topo = f"p2p_np{args.num_nodes}"
    else:
        node_keys = [d for d in PACS if d != args.leave_out]
        topo = "centralized"
    suffix = "final" if str(args.ckpt_epoch) == "final" else f"epoch_{args.ckpt_epoch}"

    rows = []
    for node in node_keys:
        ckpt = os.path.join(args.checkpoint_dir, f"{args.description}_{node}_{suffix}.pth")
        if not os.path.exists(ckpt):
            print(f"[skip] missing {ckpt}"); continue
        print(f"\n=== leave_out={args.leave_out} node={node} ({topo}, ckpt={suffix}, "
              f"6-way OSDG, unknown=idx{args.unknown_idx}) ===")
        backbone, diffusion = load_backbone_diffusion(ckpt, args.num_classes, device)

        # Test set = unseen/leave_out domain, FULL (all 7 classes incl. person=unknown).
        loader = TD.load_pacs_test_data(root, args.leave_out, args.batch_size, args.num_workers)[0]
        d_sc, msp, en, pred, lab = score_and_predict(
            backbone, diffusion, loader, diff_steps, args.ood_eval_scores_type, device)

        known = lab != args.unknown_idx
        unk = ~known
        closed_acc = float(((pred == lab) & known).sum()) / max(int(known.sum()), 1)
        print(f"  N_known={int(known.sum())} N_unknown={int(unk.sum())} | "
              f"closed-set 6-way known acc (no reject) = {closed_acc:.4f}")

        # reject_score: HIGHER = more OOD. diffusion already higher=OOD; MSP/energy higher=ID -> negate.
        for name, rej in [("diffusion", d_sc), ("msp", -msp), ("energy", -en)]:
            oscr = compute_oscr(rej, pred, lab, args.unknown_idx)
            hbest, accK, accU = compute_hscore_best(rej, pred, lab, args.unknown_idx)
            det_auroc = float(roc_auc_score(unk.astype(int), rej))   # unknown=positive=OOD
            det_aupr = float(average_precision_score(unk.astype(int), rej))  # imbalance-aware
            fpr95 = compute_fpr_at_tpr(rej, unk, 0.95)               # lower=better
            rows.append(dict(
                leave_out=args.leave_out, topo=topo, ckpt=suffix, node=node,
                unknown_idx=args.unknown_idx, score_fn=name,
                oscr=round(oscr, 4), h_best=round(hbest, 4),
                h_accK=round(accK, 4), h_accU=round(accU, 4),
                det_auroc=round(det_auroc, 4), det_aupr=round(det_aupr, 4),
                fpr95=round(fpr95, 4), closed_acc=round(closed_acc, 4),
                n_known=int(known.sum()), n_unknown=int(unk.sum())))
            print(f"  {name:9s}: OSCR={oscr:.4f}  H_best={hbest:.4f}(accK={accK:.3f},accU={accU:.3f})  "
                  f"det_AUROC={det_auroc:.4f}  AUPR={det_aupr:.4f}  FPR@95={fpr95:.4f}")

    if not rows:
        print("No rows."); return

    # node-mean summary per score_fn
    print(f"\n=== node-mean ({topo}, ckpt={suffix}, {len(node_keys)} nodes) ===")
    for name in ["diffusion", "msp", "energy"]:
        sub = [r for r in rows if r["score_fn"] == name]
        if not sub:
            continue
        mean = lambda k: float(np.mean([r[k] for r in sub]))
        print(f"  {name:9s}: OSCR={mean('oscr'):.4f}  H_best={mean('h_best'):.4f}  "
              f"det_AUROC={mean('det_auroc'):.4f}  AUPR={mean('det_aupr'):.4f}  "
              f"FPR@95={mean('fpr95'):.4f}  closed_acc={mean('closed_acc'):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    write_header = not os.path.exists(args.output_csv)
    with open(args.output_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\nAppended {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
