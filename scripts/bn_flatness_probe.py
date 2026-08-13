"""平坦度探針（A:權重空間 / B:BN統計量空間）—— 驗「async 是否找到更平的極小值」。

動機：已測 async 權重對 BN 錯配較耐受（floor/ceiling），但那不解釋成因。
領先假設＝flat minima（少融合→隱式SAM→更平）。本探針測「async 是否真的更平」：
  探針A（一般平坦、接理論）：對 sketch 權重加相對隨機噪聲、量源域(sketch) CE loss 上升。
  探針B（BN空間平坦、補「參數vs統計量」缺口）：從 cartoon-oracle-BN 出發、對 BN 統計量加
     隨機噪聲、量 cartoon acc 掉幅。
sync/async 用**相同隨機方向**（paired、公平）。async 掉得慢 ⇒ 更平 ⇒ 支持假設。

⚠ 誠實界線（見 0724§9）：A 是一般擾動、BN 錯配是結構化擾動 → 一般平坦→BN耐受是「推論非直證」；
  B 縮小此缺口（同 BN 空間）但方向仍隨機非 source→target 那個結構方向。兩支一致＝強證據鏈、非數學證明。

sanity（跑完必驗、過了才解讀）：
  (1) orig cartoon acc 重現 acc.log 末epoch；(2) σ=0/m=0 = 基準（A:Δloss=0；B:acc=oracle~83）；
  (3) 擾動單調（σ/m↑ → loss↑/acc↓）；(4) 決定性（同seed同結果）。
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as F
import util
from bn_signal_gating import build_backbone, load_backbone_only, eval_acc
from bn_recalib_probe import recompute_bn, node_files, find_res_dir
from test_domain_ood_scores import load_pacs_test_data

SETTINGS = [
    ("sync",  "exp_result_v1_stage2_leave_cartoon_seed2026_topo1234"),
    ("async", "exp_result_v1_stage2_leave_cartoon_async_const_tau1e-5_style_seed2026_topo1234"),
]
SKETCH = [6, 7, 8]
SIGMAS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]   # 探針A：相對 weight-norm
MAGS   = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]     # 探針B：BN 空間絕對 L2（域隔~29）
K = 8   # 每個 σ/m 的隨機方向數
DR = "../datasets/"


@torch.no_grad()
def source_ce_loss(model, loader):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        x, y, _ = util.unpack_batch(batch)
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        tot += F.cross_entropy(model(x), y, reduction="sum").item()
        n += x.size(0)
    return tot / n


@torch.no_grad()
def perturb_weights(model, sigma, gen):
    """對每個 param tensor 加 ‖noise‖=sigma·‖θ‖ 的隨機噪聲（相對、對不同模型公平）。回傳 originals。"""
    orig = {}
    for name, p in model.named_parameters():
        orig[name] = p.data.clone()
        if sigma > 0 and p.data.numel() > 1:
            nz = torch.randn(p.shape, generator=gen, device=p.device)
            nz = nz / (nz.norm() + 1e-12) * p.data.norm() * sigma
            p.data.add_(nz)
    return orig


@torch.no_grad()
def restore_params(model, orig):
    for name, p in model.named_parameters():
        p.data.copy_(orig[name])


@torch.no_grad()
def perturb_bn(model, mag, gen):
    """對 BN running_mean/var 加隨機噪聲、總 L2=mag；var 夾正。回傳 saved。"""
    bufs = [(n, b) for n, b in model.named_buffers()
            if n.endswith(("running_mean", "running_var"))]
    saved = {n: b.clone() for n, b in bufs}
    if mag > 0:
        nz = [torch.randn(b.shape, generator=gen, device=b.device) for _, b in bufs]
        tot = torch.sqrt(sum((x ** 2).sum() for x in nz))
        scale = mag / (tot + 1e-12)
        for (n, b), x in zip(bufs, nz):
            b.add_(x * scale)
            if n.endswith("running_var"):
                b.clamp_(min=1e-5)
    return saved


@torch.no_grad()
def restore_bufs(model, saved):
    bd = dict(model.named_buffers())
    for n, v in saved.items():
        bd[n].copy_(v)


def main():
    dev = "cuda"
    gen = torch.Generator(device=dev)
    sketch_loader, _ = load_pacs_test_data(DR, "sketch", 64, 4)
    cart_loader, _ = load_pacs_test_data(DR, "cartoon", 64, 4)

    curveA = {lab: {s: [] for s in SIGMAS} for lab, _ in SETTINGS}
    curveB = {lab: {m: [] for m in MAGS} for lab, _ in SETTINGS}
    sanity_bad = 0
    det_done = False

    for lab, ckpt_dir in SETTINGS:
        files = node_files(ckpt_dir, 9)
        res_dir = find_res_dir(ckpt_dir)
        ck0 = torch.load(files[0], map_location="cpu", weights_only=False)
        nc = ck0["backbone_state"]["backbone.fc.weight"].shape[0]
        bb = build_backbone(ck0["args"], nc, dev)

        for j in SKETCH:
            # --- sanity: orig cartoon acc vs acc.log ---
            load_backbone_only(files[j], bb, dev)
            orig_acc = eval_acc(bb, cart_loader)
            acclog = float(np.loadtxt(os.path.join(
                res_dir, f"dsgd-lr0.001-budget1.0-r{j}-acc.log"))[-1])
            ok = abs(orig_acc - acclog) < 1.5
            if not ok:
                sanity_bad += 1
            print(f"[{lab} node_{j}] orig={orig_acc:.2f} acclog={acclog:.2f} {'OK' if ok else '⚠BAD'}")

            # ===== 探針 A：權重擾動、量 sketch CE loss =====
            load_backbone_only(files[j], bb, dev)
            base_loss = source_ce_loss(bb, sketch_loader)
            for si, sigma in enumerate(SIGMAS):
                for k in range(K):
                    gen.manual_seed(1000 * j + 10 * si + k)   # sync/async 同方向（paired）
                    orig = perturb_weights(bb, sigma, gen)
                    curveA[lab][sigma].append(source_ce_loss(bb, sketch_loader) - base_loss)
                    restore_params(bb, orig)
                    if sigma == 0.0:
                        break  # σ=0 只需一次

            # ===== 探針 B：從 cartoon-oracle-BN 出發、擾動 BN、量 cartoon acc =====
            load_backbone_only(files[j], bb, dev)
            recompute_bn(bb, cart_loader, dev)             # BN = cartoon-oracle
            base_acc = eval_acc(bb, cart_loader)           # ~83（sanity）
            print(f"    [B base] cartoon-oracle acc={base_acc:.2f} (應~83)")
            for mi, mag in enumerate(MAGS):
                for k in range(K):
                    gen.manual_seed(2000 * j + 10 * mi + k)
                    saved = perturb_bn(bb, mag, gen)
                    curveB[lab][mag].append(eval_acc(bb, cart_loader))
                    restore_bufs(bb, saved)
                    if mag == 0.0:
                        break

            # --- sanity: 決定性（重跑一個擾動應一致）---
            if not det_done:
                load_backbone_only(files[j], bb, dev)
                gen.manual_seed(999); o = perturb_weights(bb, 0.1, gen); l1 = source_ce_loss(bb, sketch_loader); restore_params(bb, o)
                gen.manual_seed(999); o = perturb_weights(bb, 0.1, gen); l2 = source_ce_loss(bb, sketch_loader); restore_params(bb, o)
                print(f"    [determinism] l1={l1:.5f} l2={l2:.5f} → {'OK' if abs(l1-l2)<1e-5 else '⚠非確定'}")
                det_done = True

    # ---- 結果 ----
    print("\n================ 探針A：權重擾動 → sketch CE loss 上升 (越小越平) ================")
    print(f"{'σ':>6s} " + " ".join(f"{lab:>10s}" for lab, _ in SETTINGS) + "   ratio(async/sync)")
    for s in SIGMAS:
        a = {lab: np.mean(curveA[lab][s]) for lab, _ in SETTINGS}
        r = a["async"] / a["sync"] if a["sync"] > 1e-9 else float("nan")
        print(f"{s:6.2f} " + " ".join(f"{a[lab]:10.4f}" for lab, _ in SETTINGS) + f"   {r:.2f}")

    print("\n================ 探針B：BN擾動 → cartoon acc 掉 (掉越少越平) ================")
    print(f"{'mag':>6s} " + " ".join(f"{lab:>10s}" for lab, _ in SETTINGS) + "   Δacc(async−sync)")
    base = {lab: np.mean(curveB[lab][0.0]) for lab, _ in SETTINGS}
    for m in MAGS:
        a = {lab: np.mean(curveB[lab][m]) for lab, _ in SETTINGS}
        dd = (a["async"] - base["async"]) - (a["sync"] - base["sync"])  # async 相對掉幅 − sync
        print(f"{m:6.1f} " + " ".join(f"{a[lab]:10.2f}" for lab, _ in SETTINGS) + f"   {dd:+.2f}")

    print(f"\n[sanity] orig-vs-acclog 不符節點數={sanity_bad}（>0 則結果不可信）")
    print("判讀：探針A ratio<1 且 探針B Δacc>0（async 掉少）⇒ async 兩空間都更平 ⇒ 支持 flat-minima。")


if __name__ == "__main__":
    main()
