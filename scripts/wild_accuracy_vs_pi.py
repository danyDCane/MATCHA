"""
Wild-accuracy vs OOD-contamination (pi) — intuitive "value of having OOD detection".

Story: a DG model deployed on a WILD stream = (1-pi)*ID(target domain) + pi*far-OOD.
  - No detector (SOTA DG): must classify everything -> every OOD is a wrong/harmful output.
  - With detector (ours): can REJECT detected OOD -> abstaining counts as a CORRECT decision.

Metric: WildAcc = ( #ID-accepted-and-correct  +  #OOD-rejected ) / N
  SOTA(no reject):    WildAcc(pi) = (1-pi)*A                 [A = closed-set ID acc]
  with reject @ tau:  WildAcc(pi) = (1-pi)*a_id(tau) + pi*r_ood(tau)
      a_id(tau) = mean over ID  [ score<=tau  AND  pred==label ]   (rejected ID = lost)
      r_ood(tau)= mean over OOD [ score>tau ]                      (rejected OOD = correct)
  reject score = diffusion eps_mse (HIGHER = more OOD).

Thresholds: tpr95 / tpr99 = accept 95%/99% of ID (deployable, ID-only). oracle = per-pi
argmax_tau WildAcc (upper bound, peeks at labels — like H-score best-tau; NOT deployable).

Two stages (run infer once on GPU, then plot many times on CPU):
  --stage infer  : forward all checkpoints, save per-sample arrays to npz (the slow GPU part).
  --stage plot   : load npz, compute WildAcc lines + save figure/CSV (fast, CPU, re-runnable).

Reuses joint_eval_mixed_stream (load + score) and test_domain_ood_scores (loaders).
"""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import test_domain_ood_scores as TD
import joint_eval_mixed_stream as J

PACS = J.PACS


def _ood_loader(src, datasetRoot, bs, nw):
    if src == "textures":
        return TD.load_textures_ood_loader(datasetRoot, None, bs, nw)[0]
    if src == "svhn":
        return TD.load_svhn_ood_loader(datasetRoot, None, bs, nw)[0]
    if src == "noise":
        return TD.get_noise_loader(10000, bs, nw)
    raise ValueError(f"unknown ood source {src}")


def stage_infer(args):
    import torch  # noqa
    device = args.device if __import__("torch").cuda.is_available() else "cpu"
    diff_steps = list(range(args.num_eval_steps))
    nodes = [d for d in PACS if d != args.leave_out]
    ood_srcs = [s.strip() for s in args.ood_sources.split(",") if s.strip()]
    store = {"leave_out": args.leave_out, "nodes": np.array(nodes),
             "ood_sources": np.array(ood_srcs), "id_domain": args.leave_out}
    for nd in nodes:
        ck = os.path.join(args.checkpoint_dir, f"{args.description}_{nd}_final.pth")
        if not os.path.exists(ck):
            print(f"[skip] missing {ck}"); continue
        print(f"=== node={nd} ===")
        bb, dif = J.load_backbone_diffusion(ck, device)
        # ID = target (leave_out) domain
        id_loader = TD.load_pacs_test_data(args.datasetRoot, args.leave_out,
                                           args.batch_size, args.num_workers)[0]
        d_id, _, _, p_id, y_id = J.score_and_predict(
            bb, dif, id_loader, diff_steps, args.ood_eval_scores_type, device)
        store[f"{nd}__id_score"] = d_id.astype(np.float32)
        store[f"{nd}__id_correct"] = (p_id == y_id).astype(np.int8)
        print(f"  ID({args.leave_out} target) N={len(d_id)}  closed_acc={(p_id==y_id).mean():.4f}")
        for src in ood_srcs:
            ool = _ood_loader(src, args.datasetRoot, args.batch_size, args.num_workers)
            d_oo, _, _, _, _ = J.score_and_predict(
                bb, dif, ool, diff_steps, args.ood_eval_scores_type, device)
            store[f"{nd}__{src}__ood_score"] = d_oo.astype(np.float32)
            print(f"  OOD={src} N={len(d_oo)}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez_compressed(args.out_npz, **store)
    print(f"\nSaved per-sample arrays -> {args.out_npz}")


def _wildacc_lines(id_scores, id_correct, ood_scores, pis, tpr_list):
    """Per-node arrays in lists; returns dict of method->list(over pi) of node-mean WildAcc."""
    nN = len(id_scores)
    A = np.mean([c.mean() for c in id_correct])  # closed acc (node-mean)
    out = {"sota": [(1 - p) * A for p in pis]}
    # fixed-threshold deployable lines
    for tpr in tpr_list:
        a_id, r_oo = [], []
        for d_id, corr, d_oo in zip(id_scores, id_correct, ood_scores):
            tau = np.quantile(d_id, tpr)  # accept tpr fraction of ID
            a_id.append(((d_id <= tau) & (corr > 0)).mean())
            r_oo.append((d_oo > tau).mean())
        a_id, r_oo = np.mean(a_id), np.mean(r_oo)
        out[f"tpr{int(tpr*100)}"] = [(1 - p) * a_id + p * r_oo for p in pis]
    # oracle (per-pi argmax tau, node-mean)
    orac = []
    for p in pis:
        vals = []
        for d_id, corr, d_oo in zip(id_scores, id_correct, ood_scores):
            taus = np.unique(np.concatenate([d_id, d_oo]))
            best = max((1 - p) * ((d_id <= t) & (corr > 0)).mean() + p * (d_oo > t).mean()
                       for t in taus)
            vals.append(best)
        orac.append(np.mean(vals))
    out["oracle"] = orac
    return out, A


def stage_plot(args):
    z = np.load(args.in_npz, allow_pickle=True)
    nodes = list(z["nodes"]); leave_out = str(z["id_domain"])
    ood_srcs = list(z["ood_sources"])
    pis = np.linspace(0, args.pi_max, args.pi_points)
    tpr_list = [float(x) for x in args.tprs.split(",")]
    id_scores = [z[f"{nd}__id_score"] for nd in nodes]
    id_correct = [z[f"{nd}__id_correct"] for nd in nodes]
    rows = []
    for src in ood_srcs:
        ood_scores = [z[f"{nd}__{src}__ood_score"] for nd in nodes]
        lines, A = _wildacc_lines(id_scores, id_correct, ood_scores, pis, tpr_list)
        for i, p in enumerate(pis):
            for m, ys in lines.items():
                rows.append(dict(leave_out=leave_out, ood=src, pi=round(float(p), 4),
                                 method=m, wild_acc=round(float(ys[i]), 5)))
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.plot(pis, np.array(lines["sota"]) * 100, "o-", color="crimson",
                     label="SOTA DG (no detect)")
            for tpr in tpr_list:
                plt.plot(pis, np.array(lines[f"tpr{int(tpr*100)}"]) * 100, "s-",
                         label=f"detect @TPR{int(tpr*100)} (deployable)")
            plt.plot(pis, np.array(lines["oracle"]) * 100, "--", color="gray",
                     label="detect @oracle-tau (upper bound)")
            plt.xlabel(r"OOD contamination ratio $\pi$"); plt.ylabel("Wild accuracy (%)")
            plt.title(f"{leave_out} (det) — wild acc vs OOD ratio ({src})")
            plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
            out_png = os.path.join(args.out_dir, f"wildacc_{leave_out}_{src}.png")
            os.makedirs(args.out_dir, exist_ok=True)
            plt.savefig(out_png, dpi=130); plt.close()
            print(f"  figure -> {out_png}")
        except Exception as e:
            print(f"  (plot skip {src}: {e})")
    import csv
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"wildacc_table_{leave_out}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  table  -> {csv_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["infer", "plot"])
    # infer
    p.add_argument("--checkpoint_dir"); p.add_argument("--description")
    p.add_argument("--leave_out", choices=PACS)
    p.add_argument("--datasetRoot", default="../datasets")
    p.add_argument("--ood_sources", default="textures,svhn")
    p.add_argument("--ood_eval_scores_type", default="eps_mse")
    p.add_argument("--num_eval_steps", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_npz")
    # plot
    p.add_argument("--in_npz"); p.add_argument("--out_dir", default="results/wildacc")
    p.add_argument("--pi_max", type=float, default=0.5); p.add_argument("--pi_points", type=int, default=11)
    p.add_argument("--tprs", default="0.95,0.99")
    args = p.parse_args()
    if args.stage == "infer":
        stage_infer(args)
    else:
        stage_plot(args)


if __name__ == "__main__":
    main()
