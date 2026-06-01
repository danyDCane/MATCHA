"""Post-hoc feature dimensional-collapse analysis for V2-B-1 score-reg study.

Loads trained backbone checkpoints (V1 lambda=0 baseline OR V2-B-1 lambda=0.05),
runs a CLEAN forward (no style aug) on probe domains, collects penultimate 512-d
features, and computes dimensional-collapse metrics (effective rank / total variance
/ inter-intra ratio) via dood.score_reg_diagnostics.compute_feature_collapse_stats.

Purpose: directly test whether score-reg homogenizes the backbone representation
(plan R2, the *primary* arbiter for the collapse hypothesis — the small OOD-AUROC
delta is only suggestive). Reuses test_domain_ood_scores.load_checkpoint and the
PACSDataset loader; does NOT touch the training loop.

Usage example (one LOO, one lambda):
  venv_matcha/bin/python scripts/collapse_analysis.py \
    --leave_out art_painting \
    --checkpoint_dir exp_result_v1_stage1_leave_art_painting \
    --description v1_stage1_leave_art_painting \
    --lambda_tag 0 \
    --epochs epoch_50,epoch_100,epoch_150,epoch_200,final \
    --output_csv research/V2B1_score_norm/collapse_metrics.csv \
    --datasetRoot ../datasets/
"""

import argparse
import csv
import os
import sys
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

import util
from pacs_dataset import PACSDataset
from dood.utils.diffusion import get_diffusion_model
from test_domain_ood_scores import load_checkpoint
from dood.score_reg_diagnostics import compute_feature_collapse_stats

PACS = ["art_painting", "cartoon", "photo", "sketch"]


def build_backbone_args():
    """Minimal args namespace matching the training architecture for util.select_model."""
    return Namespace(
        model="res",
        dataset="pacs",
        resnet_type="standard",
        use_style_shift=True,
        style_shift_prob=0.8,
        style_shift_ratio=0.8,
        style_explore_alpha=3.0,
        style_explore_ratio=0.5,
        mixstyle_alpha=0.1,
        pretrained=False,  # weights overwritten by checkpoint load
    )


def get_test_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_loader(root, domain, batch_size, num_workers):
    ds = PACSDataset(root=root, dataset_name=domain, transform=get_test_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True), len(ds)


@torch.no_grad()
def extract_features(backbone, loader, device):
    """Clean forward (communicator=None + eval -> no style aug). Returns feats [N,512], labels [N]."""
    backbone.eval()
    feats, labels = [], []
    for batch in loader:
        x, y = batch[0], batch[1]  # PACSDataset returns (img, target, meta)
        x = x.to(device, non_blocking=True)
        z3 = backbone.forward_to_layer3_style(x, communicator=None)
        _, vec = backbone.forward_from_layer3(z3)
        feats.append(vec.detach().cpu())
        labels.append(y.detach().cpu().view(-1))
    return torch.cat(feats), torch.cat(labels)


def main():
    p = argparse.ArgumentParser(description="Post-hoc feature collapse analysis")
    p.add_argument("--leave_out", required=True, choices=PACS)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--description", required=True,
                   help="checkpoint filename prefix, e.g. v1_stage1_leave_art_painting")
    p.add_argument("--lambda_tag", required=True, help="label for CSV, e.g. 0 or 0.05")
    p.add_argument("--epochs", default="epoch_50,epoch_100,epoch_150,epoch_200,final",
                   help="comma list of checkpoint tags")
    p.add_argument("--include_sources", action="store_true",
                   help="also probe the 3 source domains (default: target only)")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--datasetRoot", default="../datasets/")
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--diffusion_channels", type=int, default=512)
    p.add_argument("--diffusion_steps", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    train_domains = [d for d in PACS if d != args.leave_out]
    probe_domains = [args.leave_out] + (train_domains if args.include_sources else [])
    epoch_tags = [e.strip() for e in args.epochs.split(",") if e.strip()]

    backbone = util.select_model(args.num_classes, build_backbone_args()).to(device)
    diffusion_model = get_diffusion_model(
        ft_size=512, denoiser_type="unet0d",
        diffusion_denoiser_channels=args.diffusion_channels,
        num_diffusion_steps=args.diffusion_steps,
    ).to(device)

    # cache probe features per (epoch_tag, node, probe) computed lazily
    rows = []
    for node in train_domains:  # each node = one source-domain backbone in K3
        for tag in epoch_tags:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.description}_{node}_{tag}.pth")
            if not os.path.exists(ckpt_path):
                print(f"[skip] missing {ckpt_path}")
                continue
            load_checkpoint(ckpt_path, backbone, diffusion_model, device)
            for probe in probe_domains:
                loader, n = make_loader(args.datasetRoot, probe, args.batch_size, args.num_workers)
                feats, labels = extract_features(backbone, loader, device)
                stats = compute_feature_collapse_stats(feats, labels)
                row = {
                    "leave_out": args.leave_out,
                    "lambda": args.lambda_tag,
                    "node": node,
                    "epoch": tag,
                    "probe_domain": probe,
                    "is_target": int(probe == args.leave_out),
                    "erank": round(stats["erank"], 4),
                    "tvar": round(stats["tvar"], 4),
                    "inter_intra": round(stats["inter_intra"], 4),
                    "n": stats["n_samples"],
                }
                rows.append(row)
                print(f"  {args.leave_out} lambda={args.lambda_tag} node={node} {tag} "
                      f"probe={probe}: erank={row['erank']} tvar={row['tvar']} "
                      f"inter_intra={row['inter_intra']} (n={row['n']})")

    if not rows:
        print("No rows produced (no checkpoints found).")
        return

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
