import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize spatial heatmaps from saved layer3 feature dumps.")
    parser.add_argument("--dump", type=str, required=True, help="Path to one .pt dump file or a directory of dump files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save rendered heatmaps.")
    parser.add_argument("--mode", type=str, default="l2", choices=["l2", "mean"], help="How to reduce channels into a 2D activation map.")
    parser.add_argument("--max_files", type=int, default=0, help="Maximum number of dump files to process (0 = all).")
    return parser.parse_args()


def collect_dump_files(dump_path: str):
    path = Path(dump_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Dump path not found: {dump_path}")
    files = sorted(path.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt dump files found in: {dump_path}")
    return files


def denormalize_image(tensor: torch.Tensor, mean, std):
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    img = tensor.cpu() * std_t + mean_t
    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def load_reference_image(image_path: str):
    if not image_path or not os.path.exists(image_path):
        return None
    return np.array(Image.open(image_path).convert("RGB"))


def compute_spatial_map(z: torch.Tensor, mode: str, eps: float = 1e-5):
    mu = z.mean(dim=(2, 3), keepdim=True)
    var = z.var(dim=(2, 3), unbiased=False, keepdim=True)
    z_in = (z - mu) / torch.sqrt(var + eps)
    if mode == "mean":
        heat = z_in.mean(dim=1)
    else:
        heat = torch.norm(z_in, p=2, dim=1)
    return heat


def normalize_heatmap(heatmap: torch.Tensor):
    heat = heatmap.detach().cpu().float()
    heat = heat - heat.min()
    denom = heat.max().clamp_min(1e-8)
    heat = heat / denom
    return heat.numpy()


def render_dump(dump_file: Path, output_dir: Path, mode: str):
    payload = torch.load(dump_file, map_location="cpu")
    inputs = payload["input"]
    z_clean = payload["z_clean"]
    z_style = payload["z_style"]
    z_hard = payload.get("z_hard")
    image_paths = payload.get("image_paths", [""] * len(inputs))
    mean = payload.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = payload.get("imagenet_std", [0.229, 0.224, 0.225])

    clean_maps = compute_spatial_map(z_clean, mode=mode)
    style_maps = compute_spatial_map(z_style, mode=mode)
    hard_maps = compute_spatial_map(z_hard, mode=mode) if z_hard is not None else None

    dump_output_dir = output_dir / dump_file.stem
    dump_output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(inputs.size(0)):
        panels = [
            ("Original", load_reference_image(image_paths[idx]), denormalize_image(inputs[idx], mean, std)),
            ("IN(z_clean)", None, normalize_heatmap(clean_maps[idx])),
            ("IN(z_style)", None, normalize_heatmap(style_maps[idx])),
        ]
        if hard_maps is not None:
            panels.append(("IN(z_hard)", None, normalize_heatmap(hard_maps[idx])))

        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        if len(panels) == 1:
            axes = [axes]

        for ax, (title, ref_img, rendered) in zip(axes, panels):
            if title == "Original":
                ax.imshow(ref_img if ref_img is not None else rendered)
            else:
                ax.imshow(rendered, cmap="jet")
            ax.set_title(title)
            ax.axis("off")

        basename = os.path.basename(image_paths[idx]) if image_paths[idx] else f"sample_{idx:02d}"
        fig.suptitle(f"{basename}\nstyle_aug_activated={payload.get('style_aug_activated', False)}", fontsize=11)
        fig.tight_layout()
        save_name = f"{idx:02d}_{basename}.png"
        fig.savefig(dump_output_dir / save_name, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    dump_files = collect_dump_files(args.dump)
    if args.max_files > 0:
        dump_files = dump_files[:args.max_files]

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.dump).parent / "spatial_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dump_file in dump_files:
        render_dump(dump_file, output_dir, args.mode)

    print(f"Saved heatmaps to: {output_dir}")


if __name__ == "__main__":
    main()
