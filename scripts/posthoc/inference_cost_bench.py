"""推論成本 benchmark：三組 × 三層 × 兩種 batch（dany 2026-09-07）。

三組（部署形態）：
  A  StyleDDG          只有泛化：backbone → logits → argmax
  B  StyleDDG+energy   A ＋ 拒絕分數 −logsumexp(logits)
  C  我方+原型          backbone → logits(＋vec) → project(vec) → 到 6 個類別中心的 arccos → min

三層（把「檢測多花多少」從 I/O 噪音裡分離出來）：
  (a) 端到端    讀檔 → PIL decode → Resize/ToTensor/Normalize → H2D → 前向 → 決策
  (b) 純前向    資料已在 GPU → logits（＋vec）。三組的骨幹相同，此層應幾乎一致
  (c) 讀出      已有 logits/vec → 拒絕分數。★ 這一層才是「檢測的代價」

⚠️ 歸屬約定：`project()`（512→128 MLP）算在 (c) 讀出，因為它只服務檢測——
   分類走 `backbone.fc`，不經過它。這個歸屬讓 (c) 直接回答「多了檢測慢多少」。

⚠️ 計時紀律：每段前後 `torch.cuda.synchronize()`（GPU 非同步，不同步量到的是送指令的時間）；
   warmup 50 次（首次前向含 CUDA context 初始化與 cudnn autotune，會慢數倍）；
   取 median 與 p95，不用單次值。

⚠️ BN 平均（我方的「平均B」口徑）**不影響推論時間**——它只改 buffer 的數值、不改計算量，
   且是部署前一次性操作。它在去中心設定下的成本是一次全域統計匯聚（通訊），不在本表。

⚠️ 端到端的檔案讀取受 OS page cache 影響：本腳本先完整讀過一輪再計時（熱快取），
   對應「圖片已在本機」的部署形態；冷啟動（首次從磁碟）會更慢，不在本表射程。
"""
import os
import sys
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    sys.path.insert(0, _p)

import torch
from PIL import Image
from torchvision import transforms
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers

TF = transforms.Compose([                     # 與 test_domain_ood_scores.load_pacs_test_data 逐項相同
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def timed(fn, n, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)      # ms
    a = np.array(ts)
    return dict(median=float(np.median(a)), mean=float(a.mean()),
                p95=float(np.percentile(a, 95)), std=float(a.std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leave_out", default="cartoon")
    ap.add_argument("--datasetRoot", default="../datasets/")
    ap.add_argument("--n_files", type=int, default=64)
    ap.add_argument("--reps_b1", type=int, default=300)
    ap.add_argument("--reps_b64", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    LO = a.leave_out
    B_DESC = f"v1_stage2_leave_{LO}_nodiff_osdg_excl_person_seed2026_topo1234"
    O_DESC = (f"v1_stage2_leave_{LO}_p1a_async_const_tau1e-5_style_nodiff_aggbn_"
              f"osdg_excl_person_seed2026_topo1234") + ("_fix" if LO == "cartoon" else "")
    bb_b, _ = load_backbone_diffusion(
        os.path.join(ROOT, f"exp_result_{B_DESC}", f"{B_DESC}_node_0_final.pth"), 6, dev)
    bb_o, _ = load_backbone_diffusion(
        os.path.join(ROOT, f"exp_result_{O_DESC}", f"{O_DESC}_node_0_final.pth"), 6, dev)
    bb_b.eval(); bb_o.eval()
    C = class_centers(bb_o.prototypes, bb_o.proto_count).to(dev)

    # 蒐集真實圖片路徑（端到端層要真的讀檔）
    root = os.path.join(a.datasetRoot, "PACS", LO)
    if not os.path.isdir(root):
        root = os.path.join(a.datasetRoot, "PACS", "kfold", LO)
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                files.append(os.path.join(dp, fn))
    files = sorted(files)[:a.n_files]
    if not files:
        raise FileNotFoundError(f"找不到圖片：{root}")
    for f in files:                                   # 預熱 OS page cache
        with open(f, "rb") as fh:
            fh.read()

    print(f"環境：{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"fold={LO} | 圖片 {len(files)} 張 | warmup={a.warmup}")
    print(f"參數量：StyleDDG {sum(p.numel() for p in bb_b.parameters()):,} | "
          f"我方 {sum(p.numel() for p in bb_o.parameters()):,}")

    @torch.no_grad()
    def run(bs, reps):
        x = torch.randn(bs, 3, 224, 224, device=dev)
        out = {}
        # ---- (b) 純前向 ----
        def fwd_b():
            z3 = bb_b.forward_to_layer3_style(x, communicator=None)
            return bb_b.forward_from_layer3(z3)
        def fwd_o():
            z3 = bb_o.forward_to_layer3_style(x, communicator=None)
            return bb_o.forward_from_layer3(z3)
        out[("b", "A")] = timed(lambda: fwd_b(), reps, a.warmup)
        out[("b", "B")] = out[("b", "A")]                      # B 與 A 同一次前向
        out[("b", "C")] = timed(lambda: fwd_o(), reps, a.warmup)
        # ---- (c) 讀出（用已算好的 logits/vec，只量讀出本身）----
        lo_b, _ = fwd_b()
        lo_o, ve_o = fwd_o()
        out[("c", "A")] = timed(lambda: lo_b.argmax(1), reps, a.warmup)
        out[("c", "B")] = timed(lambda: (lo_b.argmax(1), -torch.logsumexp(lo_b, 1)), reps, a.warmup)
        def readout_c():
            z = bb_o.project(ve_o)
            cos = (z @ C.t()).clamp(-1 + 1e-7, 1 - 1e-7)
            return lo_o.argmax(1), torch.arccos(cos).min(1).values
        out[("c", "C")] = timed(readout_c, reps, a.warmup)
        # ---- (a) 端到端（含讀檔與前處理）----
        def e2e(bb, mode):
            batch = torch.stack([TF(Image.open(files[i % len(files)]).convert("RGB"))
                                 for i in range(bs)]).to(dev, non_blocking=True)
            z3 = bb.forward_to_layer3_style(batch, communicator=None)
            lo, ve = bb.forward_from_layer3(z3)
            if mode == "A":
                return lo.argmax(1)
            if mode == "B":
                return lo.argmax(1), -torch.logsumexp(lo, 1)
            z = bb.project(ve)
            cos = (z @ C.t()).clamp(-1 + 1e-7, 1 - 1e-7)
            return lo.argmax(1), torch.arccos(cos).min(1).values
        e2e_reps = max(20, reps // 10)                          # 端到端含 CPU 解碼，次數少一點
        for m, bb in [("A", bb_b), ("B", bb_b), ("C", bb_o)]:
            out[("a", m)] = timed(lambda bb=bb, m=m: e2e(bb, m), e2e_reps, min(10, a.warmup))
        return out

    NAME = {"A": "StyleDDG（只有泛化）", "B": "StyleDDG+energy", "C": "我方+原型"}
    LAYER = {"a": "(a) 端到端 含讀檔前處理", "b": "(b) 純模型前向", "c": "(c) 讀出本身 ★檢測代價"}
    for bs, reps in [(1, a.reps_b1), (64, a.reps_b64)]:
        R = run(bs, reps)
        print("\n" + "=" * 96)
        print(f"batch size = {bs}（{'單張延遲' if bs == 1 else '批次吞吐'}）")
        print("=" * 96)
        for layer in ["a", "b", "c"]:
            print(f"\n  {LAYER[layer]}")
            print(f"    {'組':<22}{'median(ms)':>12}{'mean(ms)':>11}{'p95(ms)':>10}{'std':>8}"
                  + (f"{'img/s':>10}" if bs > 1 else ""))
            base = R[(layer, "A")]["median"]
            for m in ["A", "B", "C"]:
                r = R[(layer, m)]
                thr = f"{bs / r['median'] * 1000:>10.0f}" if bs > 1 else ""
                d = "" if m == "A" else f"   ({r['median']-base:+.4f} ms, {(r['median']/base-1)*100:+.1f}%)"
                print(f"    {NAME[m]:<22}{r['median']:>12.4f}{r['mean']:>11.4f}"
                      f"{r['p95']:>10.4f}{r['std']:>8.4f}{thr}{d}")
        print(f"\n  ★ 檢測的淨代價（(c) 層相對 A）：energy {R[('c','B')]['median']-R[('c','A')]['median']:+.4f} ms"
              f"／原型 {R[('c','C')]['median']-R[('c','A')]['median']:+.4f} ms")
        tot_a, tot_c = R[("a", "A")]["median"], R[("a", "C")]["median"]
        print(f"  ★ 端到端相對增幅：energy {(R[('a','B')]['median']/tot_a-1)*100:+.2f}%"
              f"／原型 {(tot_c/tot_a-1)*100:+.2f}%")


if __name__ == "__main__":
    main()
