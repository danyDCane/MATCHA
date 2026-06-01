"""
Post-hoc "what does the classifier rely on" diagnostic on trained det backbones.

Reuses checkpoint loading / dataloaders / clean feature extraction from
scripts/collapse_analysis.py (no retraining, no loader rewrite).

Probes (select via --probes p1,p2,p3):
  P1  style/domain residual : linear probe on frozen vec_style predicting
        domain (4-way) vs class (7-way). High domain acc = style retained.
  P2  texture-vs-shape       : target/source accuracy on original / low-pass
        (shape) / high-pass (texture) frequency-filtered images.
  P3  style invariance        : clean vs style-perturbed forward consistency
        (sym_kl / ce_cos / prediction-flip) on the held-out target. Style here =
        StyleExplore + MixStyle (within-batch, no communicator; StyleShift's
        cross-node exchange is omitted -- see report caveat).

Usage (one LOO, one lambda):
    venv_matcha/bin/python scripts/feature_reliance_probe.py \
        --leave_out art_painting \
        --checkpoint_dir exp_result_v1_stage1_leave_art_painting \
        --description v1_stage1_leave_art_painting \
        --lambda_tag 0 \
        --probes p1,p2,p3 \
        --output_csv research/V2B1_score_norm/0529_feature_reliance_probe.csv \
        --datasetRoot /home/server5090/Desktop/M11307320/datasets
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import util
import collapse_analysis as CA  # reuse build_backbone_args / make_loader / extract_features / load_checkpoint / get_diffusion_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from style_transforms import adain  # reuse exact AdaIN used in training
from style_stats import compute_layer_style_stats  # reuse exact mu_bar/sigma_bar

PACS = CA.PACS
SEED = 1234
P3_LAYERS = ["layer1", "layer2", "layer3"]


def load_backbone(ckpt_path, device):
    backbone = util.select_model(7, CA.build_backbone_args()).to(device)
    diffusion = CA.get_diffusion_model(
        ft_size=512, denoiser_type="unet0d",
        diffusion_denoiser_channels=512, num_diffusion_steps=1000,
    ).to(device)
    CA.load_checkpoint(ckpt_path, backbone, diffusion, device)
    backbone.eval()
    return backbone


# ----------------------------------------------------------------------------- P1
def probe_p1(backbone, root, device, bs, nw):
    feats, ys_cls, ys_dom = [], [], []
    for di, dom in enumerate(PACS):
        loader, _ = CA.make_loader(root, dom, bs, nw)
        f, y = CA.extract_features(backbone, loader, device)  # clean forward, vec [N,512]
        feats.append(f.numpy())
        ys_cls.append(y.numpy())
        ys_dom.append(np.full(len(y), di))
    X = np.concatenate(feats)
    y_cls = np.concatenate(ys_cls)
    y_dom = np.concatenate(ys_dom)

    def linprobe(y):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(Xtr), ytr)
        return float((clf.predict(sc.transform(Xte)) == yte).mean())

    dom_chance = float(np.bincount(y_dom).max() / len(y_dom))  # majority-class baseline
    return {
        "p1_domain_probe_acc": round(linprobe(y_dom), 4),
        "p1_domain_chance": round(dom_chance, 4),
        "p1_class_probe_acc": round(linprobe(y_cls), 4),
    }


# ----------------------------------------------------------------------------- P2
def _radial_mask(H, W, cutoff, low, device):
    fy = torch.fft.fftfreq(H, device=device).view(H, 1)
    fx = torch.fft.fftfreq(W, device=device).view(1, W)
    r = torch.sqrt(fy ** 2 + fx ** 2)
    rn = r / r.max()
    return (rn <= cutoff) if low else (rn > cutoff)


def _freq_filter(x, cutoff, low):
    # x: [B,C,H,W] in [0,1]
    H, W = x.shape[-2:]
    X = torch.fft.fft2(x)
    m = _radial_mask(H, W, cutoff, low, x.device).to(X.dtype)
    xf = torch.fft.ifft2(X * m).real
    return xf.clamp(0.0, 1.0)


_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _raw_loader(root, domain, bs, nw):
    from torchvision import transforms
    from pacs_dataset import PACSDataset
    from torch.utils.data import DataLoader
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])  # [0,1], no normalize
    ds = PACSDataset(root=root, dataset_name=domain, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)


@torch.no_grad()
def _acc_on(backbone, loader, device, cutoff, mode):
    mean = _NORM_MEAN.to(device)
    std = _NORM_STD.to(device)
    correct = total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device).view(-1)
        if mode == "low":
            x = _freq_filter(x, cutoff, low=True)
        elif mode == "high":
            x = _freq_filter(x, cutoff, low=False)
        x = (x - mean) / std
        z3 = backbone.forward_to_layer3_style(x, communicator=None)
        logits, _ = backbone.forward_from_layer3(z3)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def probe_p2(backbone, root, leave_out, device, bs, nw, cutoff):
    sources = [d for d in PACS if d != leave_out]
    out = {}
    tl = _raw_loader(root, leave_out, bs, nw)
    out["p2_target_acc_orig"] = round(_acc_on(backbone, tl, device, cutoff, "orig"), 4)
    out["p2_target_acc_low"] = round(_acc_on(backbone, tl, device, cutoff, "low"), 4)
    out["p2_target_acc_high"] = round(_acc_on(backbone, tl, device, cutoff, "high"), 4)
    so, sl, sh = [], [], []
    for s in sources:
        lo_ = _raw_loader(root, s, bs, nw)
        so.append(_acc_on(backbone, lo_, device, cutoff, "orig"))
        sl.append(_acc_on(backbone, lo_, device, cutoff, "low"))
        sh.append(_acc_on(backbone, lo_, device, cutoff, "high"))
    out["p2_source_acc_orig"] = round(float(np.mean(so)), 4)
    out["p2_source_acc_low"] = round(float(np.mean(sl)), 4)
    out["p2_source_acc_high"] = round(float(np.mean(sh)), 4)
    out["p2_cutoff"] = cutoff
    return out


# ----------------------------------------------------------------------------- P3
# Style-counterfactual: restyle content features to a REAL source domain's mean style
# (deterministic AdaIN at layer1/2/3, no DSU / no explore / no mix). Report BOTH
# accuracy (style problem vs content problem) and consistency (flip/sym_kl).
def _forward_capture(b, x):
    h = b.backbone.conv1(x); h = b.backbone.bn1(h); h = b.backbone.relu(h); h = b.backbone.maxpool(h)
    o1 = b.backbone.layer1(h)
    o2 = b.backbone.layer2(o1)
    o3 = b.backbone.layer3(o2)
    return o1, o2, o3


def _restyle_logits(b, x, style, apply_layers):
    # style: {layer_name: (mu_bar[C], sigma_bar[C])}; apply_layers: subset of P3_LAYERS
    h = b.backbone.conv1(x); h = b.backbone.bn1(h); h = b.backbone.relu(h); h = b.backbone.maxpool(h)
    o1 = b.backbone.layer1(h)
    if "layer1" in apply_layers:
        o1 = adain(o1, style["layer1"][0], style["layer1"][1])
    o2 = b.backbone.layer2(o1)
    if "layer2" in apply_layers:
        o2 = adain(o2, style["layer2"][0], style["layer2"][1])
    o3 = b.backbone.layer3(o2)
    if "layer3" in apply_layers:
        o3 = adain(o3, style["layer3"][0], style["layer3"][1])
    logits, _ = b.forward_from_layer3(o3)
    return logits


@torch.no_grad()
def _domain_style(b, loader, device):
    # domain-level (mu_bar, sigma_bar) per layer = per-sample channel mean/std averaged over domain
    sums = {L: [None, None] for L in P3_LAYERS}
    n = 0
    for batch in loader:
        x = batch[0].to(device)
        outs = dict(zip(P3_LAYERS, _forward_capture(b, x)))
        bs = x.size(0); n += bs
        for L in P3_LAYERS:
            st = compute_layer_style_stats(outs[L])  # mu_bar/sigma_bar are batch means
            mu, sig = st["mu_bar"] * bs, st["sigma_bar"] * bs
            sums[L][0] = mu if sums[L][0] is None else sums[L][0] + mu
            sums[L][1] = sig if sums[L][1] is None else sums[L][1] + sig
    return {L: (sums[L][0] / n, sums[L][1] / n) for L in P3_LAYERS}


@torch.no_grad()
def _eval_restyle(b, loader, style, apply_layers, device):
    eps = 1e-8
    correct = raw_correct = flip = tot = 0
    symkls = []
    for batch in loader:
        x = batch[0].to(device); y = batch[1].to(device).view(-1)
        clean = _restyle_logits(b, x, style, [])
        styled = _restyle_logits(b, x, style, apply_layers)
        raw_correct += (clean.argmax(1) == y).sum().item()
        correct += (styled.argmax(1) == y).sum().item()
        flip += (clean.argmax(1) != styled.argmax(1)).sum().item()
        pc = F.softmax(clean, 1); ps = F.softmax(styled, 1)
        kl_cs = F.kl_div(torch.log(pc.clamp_min(eps)), ps, reduction="batchmean")
        kl_sc = F.kl_div(torch.log(ps.clamp_min(eps)), pc, reduction="batchmean")
        symkls.append(float(0.5 * (kl_cs + kl_sc)))
        tot += y.numel()
    return correct / tot, flip / tot, float(np.mean(symkls)), raw_correct / tot


def probe_p3(backbone, root, leave_out, device, bs, nw):
    sources = [d for d in PACS if d != leave_out]
    styles = {d: _domain_style(backbone, CA.make_loader(root, d, bs, nw)[0], device) for d in PACS}
    ablations = {"res1": ["layer1"], "res3": ["layer3"], "res123": P3_LAYERS}
    out = {}

    # TARGET content restyled to each REAL source style, averaged over sources
    raw_ref = None
    for ab, layers in ablations.items():
        accs, flips, kls, raws = [], [], [], []
        for S in sources:
            acc, flip, kl, raw = _eval_restyle(
                backbone, CA.make_loader(root, leave_out, bs, nw)[0], styles[S], layers, device)
            accs.append(acc); flips.append(flip); kls.append(kl); raws.append(raw)
        out[f"p3_target_acc_{ab}"] = round(float(np.mean(accs)), 4)
        out[f"p3_target_flip_{ab}"] = round(float(np.mean(flips)), 4)
        out[f"p3_target_symkl_{ab}"] = round(float(np.mean(kls)), 6)
        raw_ref = float(np.mean(raws))
    out["p3_target_raw_acc"] = round(raw_ref, 4)

    # SOURCE control: source content restyled to ANOTHER source style (res123)
    sc_acc, sc_flip, sc_raw = [], [], []
    for s in sources:
        for S in [o for o in sources if o != s]:
            acc, flip, _, raw = _eval_restyle(
                backbone, CA.make_loader(root, s, bs, nw)[0], styles[S], P3_LAYERS, device)
            sc_acc.append(acc); sc_flip.append(flip); sc_raw.append(raw)
    out["p3_srcctrl_acc_res123"] = round(float(np.mean(sc_acc)), 4)
    out["p3_srcctrl_flip_res123"] = round(float(np.mean(sc_flip)), 4)
    out["p3_srcctrl_raw_acc"] = round(float(np.mean(sc_raw)), 4)
    return out


# ----------------------------------------------------------------------------- P4
# Image-amplitude counterfactual (FOOGD-faithful): each target image is paired with a
# RANDOM REAL source-domain image; we swap amplitude (lambda=1 full) keeping target's
# phase, then forward. Matches FOOGD `colorful_spectrum_mix` (data_utils.py:61).
# Stochasticity over partners handled by N seeds and averaging metrics.
# Predecessor (domain-mean amplitude) was retired: produced image-space chimera
# (mean of |fft| has no real-image counterpart) -> src-control also dropped large -> couldn't attribute target effect to "style".
def _shuffled_raw_loader(root, domain, bs, nw, gen):
    from torchvision import transforms
    from pacs_dataset import PACSDataset
    from torch.utils.data import DataLoader
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    ds = PACSDataset(root=root, dataset_name=domain, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, generator=gen)


def _pair_amp_swap(x_t, x_p, lam, ratio):
    # x_t (target), x_p (partner): [B,C,H,W] in [0,1]; lam in [0,1]; ratio in (0,1]
    H, W = x_t.shape[-2:]
    F_t = torch.fft.fft2(x_t, dim=(-2, -1))
    F_p = torch.fft.fft2(x_p, dim=(-2, -1))
    abs_t = torch.abs(F_t); ang_t = torch.angle(F_t)
    abs_p = torch.abs(F_p)
    abs_t_s = torch.fft.fftshift(abs_t, dim=(-2, -1))
    abs_p_s = torch.fft.fftshift(abs_p, dim=(-2, -1))
    if ratio >= 1.0:
        abs_new_s = (1 - lam) * abs_t_s + lam * abs_p_s
    else:
        h_crop = int(H * math.sqrt(ratio)); w_crop = int(W * math.sqrt(ratio))
        h_st = H // 2 - h_crop // 2; w_st = W // 2 - w_crop // 2
        abs_new_s = abs_t_s.clone()
        abs_new_s[..., h_st:h_st + h_crop, w_st:w_st + w_crop] = (
            (1 - lam) * abs_t_s[..., h_st:h_st + h_crop, w_st:w_st + w_crop]
            + lam * abs_p_s[..., h_st:h_st + h_crop, w_st:w_st + w_crop]
        )
    abs_new = torch.fft.ifftshift(abs_new_s, dim=(-2, -1))
    F_new = abs_new * torch.exp(1j * ang_t)
    x_new = torch.real(torch.fft.ifft2(F_new, dim=(-2, -1)))
    return x_new.clamp(0.0, 1.0)


@torch.no_grad()
def _eval_paired(b, target_factory, partner_factory, lam, ratio, device, n_seeds):
    mean = _NORM_MEAN.to(device); std = _NORM_STD.to(device)
    seed_acc, seed_flip, seed_raw = [], [], []
    for s in range(n_seeds):
        g = torch.Generator(); g.manual_seed(SEED + s)
        tloader = target_factory()
        ploader = partner_factory(g)
        correct = raw_correct = flip = tot = 0
        for tb, pb in zip(tloader, ploader):
            x_t = tb[0].to(device); y = tb[1].to(device).view(-1)
            x_p = pb[0].to(device)
            n = min(x_t.size(0), x_p.size(0))
            if n < x_t.size(0): x_t, y = x_t[:n], y[:n]
            if n < x_p.size(0): x_p = x_p[:n]
            x_styled = _pair_amp_swap(x_t, x_p, lam, ratio)
            xc = (x_t - mean) / std; xs = (x_styled - mean) / std
            z3c = b.forward_to_layer3_style(xc, communicator=None)
            lc, _ = b.forward_from_layer3(z3c)
            z3s = b.forward_to_layer3_style(xs, communicator=None)
            ls, _ = b.forward_from_layer3(z3s)
            raw_correct += (lc.argmax(1) == y).sum().item()
            correct += (ls.argmax(1) == y).sum().item()
            flip += (lc.argmax(1) != ls.argmax(1)).sum().item()
            tot += y.numel()
        seed_acc.append(correct / tot); seed_flip.append(flip / tot); seed_raw.append(raw_correct / tot)
    return float(np.mean(seed_acc)), float(np.mean(seed_flip)), float(np.mean(seed_raw))


def probe_p4(backbone, root, leave_out, device, bs, nw):
    sources = [d for d in PACS if d != leave_out]
    out = {}
    LAM = 1.0
    N_SEEDS = 5

    # TARGET phase x random real source-S image's amplitude (FOOGD-faithful);
    # ratios = 1.0 (full amp swap) and 0.1 (central low-freq amp swap)
    for tag, ratio in [("r1", 1.0), ("r01", 0.1)]:
        accs, flips, raws = [], [], []
        for S in sources:
            acc, flip, raw = _eval_paired(
                backbone,
                lambda _lo=leave_out: _raw_loader(root, _lo, bs, nw),
                lambda g, _S=S: _shuffled_raw_loader(root, _S, bs, nw, g),
                LAM, ratio, device, N_SEEDS)
            accs.append(acc); flips.append(flip); raws.append(raw)
        out[f"p4_target_acc_lam1_{tag}"] = round(float(np.mean(accs)), 4)
        out[f"p4_target_flip_lam1_{tag}"] = round(float(np.mean(flips)), 4)
        out["p4_target_raw_acc"] = round(float(np.mean(raws)), 4)

    # SRC-CONTROL: source-s phase x random other-source image's amplitude (ratio=1)
    sc_acc, sc_flip, sc_raw = [], [], []
    for s in sources:
        for S in [o for o in sources if o != s]:
            acc, flip, raw = _eval_paired(
                backbone,
                lambda _s=s: _raw_loader(root, _s, bs, nw),
                lambda g, _S=S: _shuffled_raw_loader(root, _S, bs, nw, g),
                LAM, 1.0, device, N_SEEDS)
            sc_acc.append(acc); sc_flip.append(flip); sc_raw.append(raw)
    out["p4_srcctrl_acc_lam1_r1"] = round(float(np.mean(sc_acc)), 4)
    out["p4_srcctrl_flip_lam1_r1"] = round(float(np.mean(sc_flip)), 4)
    out["p4_srcctrl_raw_acc"] = round(float(np.mean(sc_raw)), 4)
    return out


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leave_out", required=True, choices=PACS)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--lambda_tag", required=True)
    p.add_argument("--probes", default="p1,p2,p3")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--datasetRoot", default="../datasets/")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--p2_cutoff", type=float, default=0.15)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    probes = [s.strip() for s in args.probes.split(",") if s.strip()]
    nodes = [d for d in PACS if d != args.leave_out]

    rows = []
    for node in nodes:
        ckpt = os.path.join(args.checkpoint_dir, f"{args.description}_{node}_final.pth")
        if not os.path.exists(ckpt):
            print(f"[skip] missing {ckpt}")
            continue
        print(f"\n=== {args.leave_out} lambda={args.lambda_tag} node={node} ===")
        backbone = load_backbone(ckpt, device)
        metrics = {}
        if "p1" in probes:
            metrics.update(probe_p1(backbone, args.datasetRoot, device, args.batch_size, args.num_workers))
        if "p2" in probes:
            metrics.update(probe_p2(backbone, args.datasetRoot, args.leave_out, device,
                                    args.batch_size, args.num_workers, args.p2_cutoff))
        if "p3" in probes:
            metrics.update(probe_p3(backbone, args.datasetRoot, args.leave_out, device,
                                    args.batch_size, args.num_workers))
        if "p4" in probes:
            metrics.update(probe_p4(backbone, args.datasetRoot, args.leave_out, device,
                                    args.batch_size, args.num_workers))
        for k, v in metrics.items():
            print(f"  {k} = {v}")
            rows.append({"leave_out": args.leave_out, "lambda": args.lambda_tag,
                         "node": node, "metric": k, "value": v})

    if not rows:
        print("No rows produced.")
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    write_header = not os.path.exists(args.output_csv)
    with open(args.output_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["leave_out", "lambda", "node", "metric", "value"])
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\nAppended {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
