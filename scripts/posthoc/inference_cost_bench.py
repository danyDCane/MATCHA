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


@torch.no_grad()
def decompose(bb_b, bb_o, C, dev, a):
    """拆解「讀出層」的耗時：是算術量、還是叫指令的固定成本？（dany 2026-09-22 提問）

    ★ 口徑（2026-09-23 統一，與本檔 §8.3 三層表逐字一致，**不得再出現第二套**）：
      - 三個臂都含 `argmax`（A＝只分類｜B＝分類＋energy｜C＝分類＋原型讀出），
        因此「淨代價」＝該臂 − A 臂，與 0905 §8.3 括號裡的數字同一個算法。
      - **量測順序寫死：牆鐘 → CPU 送指令 → profiler**。profiler(CUPTI) 掛著會墊高後續計時，
        2026-09-22 因為把 profiler 跑在前面，量到的牆鐘比正確值高 15~20%（C 71.2 vs 58.7 µs）。

    三個量的定義（三者不可互相取代）：
      牆鐘      前後各 synchronize ⇒ 「要求答案 → 拿到答案」的實際等待
      CPU送指令 連續呼叫 N 次、最後才 synchronize ⇒ 只算把指令送出去的成本（不等 GPU）
      GPU執行   profiler 的 device_time_total ⇒ 顯示卡真正在算的時間
    ⚠️ 牆鐘 ≠ CPU送指令 ＋ GPU執行：兩者重疊（CPU 送第二道時 GPU 已在算第一道）
       ⇒ 牆鐘 ≈ max(兩者) ＋ 同步成本。本設定下 CPU 較慢 ⇒ 牆鐘 ≈ CPU送指令 ＋ ~5µs。

    ⚠️ 臂 A1 的等價性：`min_c arccos(cos)` ＝ `arccos(max_c cos)`，arccos 嚴格遞減
       ⇒ 用 `−max_c cos` 與原式**排名逐位相同**、AUROC 不變（實測 9 節點差 0.00e+00）。
    """
    W1, W2 = bb_o.proj_head[0], bb_o.proj_head[2]
    batches = [int(s_) for s_ in a.decompose_batches.split(",")]

    mac_proj = W1.in_features * W1.out_features + W2.in_features * W2.out_features
    print("=" * 96)
    print(f"環境：{torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    print("\n【算術量／每張圖】骨幹 ResNet-18 224² ≈ 1,814 MMAC（文獻標準值，非本腳本量測）｜"
          f"proj_head {mac_proj/1e6:.3f} MMAC｜z·C {C.shape[0]*C.shape[1]/1e6:.6f} MMAC")
    print(f"  ⇒ 我方讀出佔骨幹算術量 {(mac_proj + C.shape[0]*C.shape[1])/1.814e9*100:.3f}%")

    def _wall(fn, reps, warm):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)
        return float(np.median(ts))

    def _disp(fn, reps, warm):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        d = (time.perf_counter() - t0) / reps * 1e6
        torch.cuda.synchronize()
        return d

    for bs in batches:
        x = torch.randn(bs, 3, 224, 224, device=dev)
        lo, ve = bb_o.forward_from_layer3(bb_o.forward_to_layer3_style(x, communicator=None))
        reps = a.reps_b1 if bs == 1 else a.reps_b64

        def arm_A():
            return lo.argmax(1)

        def arm_B():
            return lo.argmax(1), -torch.logsumexp(lo, 1)

        def arm_C():                       # 現行 dood/prototype.detection_score
            z = bb_o.project(ve)
            c = (z @ C.t()).clamp(-1 + 1e-7, 1 - 1e-7)
            return lo.argmax(1), torch.arccos(c).min(1).values

        def arm_C1():                      # A1：−max cos，排名逐位相同
            u = W2(torch.relu(W1(ve)))
            return lo.argmax(1), -(u @ C.t()).max(1).values / u.norm(dim=1)

        ARMS = [("A 只做分類", arm_A), ("B 分類＋energy", arm_B),
                ("C 分類＋我方現行讀出", arm_C), ("C1 分類＋`−max cos` 等價改寫", arm_C1)]

        wall = {n: _wall(f, reps, a.warmup) for n, f in ARMS}           # ① 牆鐘最先
        disp = {n: _disp(f, max(reps, 1000), a.warmup) for n, f in ARMS}  # ② 再送指令
        from torch.profiler import profile, ProfilerActivity            # ③ profiler 最後
        gpu, nk = {}, {}
        for n, f in ARMS:
            for _ in range(a.warmup):
                f()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as pr:
                for _ in range(100):
                    f()
                torch.cuda.synchronize()
            r = [e for e in pr.key_averages() if e.device_time_total > 0]
            gpu[n] = sum(e.device_time_total for e in r) / 100
            nk[n] = sum(e.count for e in r) / 100

        print("\n" + "=" * 96)
        print(f"batch = {bs}　（口徑：三臂皆含 argmax，同 §8.3；牆鐘先量、profiler 最後）")
        print("=" * 96)
        print(f"  {'臂':<30}{'指令道數':>9}{'CPU送指令':>12}{'GPU執行':>11}{'牆鐘':>10}")
        for n, _ in ARMS:
            print(f"  {n:<30}{nk[n]:>9.0f}{disp[n]:>11.1f}µs{gpu[n]:>10.1f}µs{wall[n]:>9.1f}µs")

        A_, B_, Cn, C1 = [n for n, _ in ARMS]
        print(f"\n  淨代價（牆鐘 − A 臂牆鐘，＝§8.3 括號的算法）：")
        for n in (B_, Cn, C1):
            print(f"    {n:<30}+{wall[n]-wall[A_]:6.1f}µs")
        print(f"    ⇒ C ÷ B ＝ {(wall[Cn]-wall[A_])/(wall[B_]-wall[A_]):.2f}×"
              f"（＝教授看到的『快兩倍』）")

        dd, dg, dw = disp[Cn]-disp[B_], gpu[Cn]-gpu[B_], wall[Cn]-wall[B_]
        uB, uC = disp[B_]/max(nk[B_], 1), disp[Cn]/max(nk[Cn], 1)
        print(f"\n  C − B（我方比 energy 多花的）：道數 +{nk[Cn]-nk[B_]:.0f}　"
              f"CPU送指令 +{dd:.1f}µs　GPU執行 +{dg:.1f}µs　牆鐘 +{dw:.1f}µs")
        print(f"    ★ 牆鐘差 {dw:.1f} ≈ CPU送指令差 {dd:.1f} ⇒ 差距幾乎全在『送指令』，"
              f"GPU 多算的 {dg:.1f}µs 被藏在等待裡")
        print(f"    拆 CPU送指令差 {dd:.1f}µs：單價 B {uB:.2f} vs C {uC:.2f} µs/道")
        print(f"      多出的 {nk[Cn]-nk[B_]:.0f} 道 × B 單價 ＝ {(nk[Cn]-nk[B_])*uB:.1f}µs")
        print(f"      C 的 {nk[Cn]:.0f} 道 × 單價差 {uC-uB:.2f} ＝ {(uC-uB)*nk[Cn]:.1f}µs")

        print(f"\n  單一指令的送出成本（解釋單價差的來源）：")
        aa, bb_ = torch.randn(bs, 512, device=dev), torch.randn(bs, 512, device=dev)
        for nm, f in [("逐元素加法 a+b", lambda: aa + bb_),
                      ("逐元素 exp(a)", lambda: torch.exp(aa)),
                      ("矩陣乘法 Linear(512→512)", lambda: W1(aa)),
                      ("矩陣乘法 Linear(512→128)", lambda: W2(aa))]:
            print(f"    {nm:<34}{_disp(f, 2000, 200):>7.2f}µs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leave_out", default="cartoon")
    ap.add_argument("--datasetRoot", default="../datasets/")
    ap.add_argument("--n_files", type=int, default=64)
    ap.add_argument("--reps_b1", type=int, default=300)
    ap.add_argument("--reps_b64", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--decompose", action="store_true",
                    help="拆解 (c) 讀出層：逐 op 計時 ＋ 核心啟動次數 ＋ 算術量 ＋ 四個等價改寫臂。"
                         "不帶此旗標時行為與原本完全相同（dany 2026-09-22）。")
    ap.add_argument("--decompose_batches", default="1,8,64,256")
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

    if a.decompose:
        decompose(bb_b, bb_o, C, dev, a)
        return

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
