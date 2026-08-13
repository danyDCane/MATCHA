"""BN 錯配位移的「幾何」探針 —— 直接答「為何 async 決策函數扛等量位移」。

已知：async 的 source→cartoon 特徵位移量沒更小（BN 距離~29 both），但傷分類傷得少（地板高）。
數學上只可能是：
  假設1 位移方向更「class-irrelevant」：Δfeat 落在分類器零空間（垂直於決策邊界）→ 不動 logit。
  假設2 margin 更大：邊界離樣本更遠、同位移推不過去。

測（sketch 節點、target=cartoon、對每張圖）：
  Δfeat = feat(source-BN) − feat(cartoon-BN)         （512維、fc 前）
  frac_class = ‖proj_{分類器子空間}(Δfeat)‖² / ‖Δfeat‖²
    分類器子空間 = 中心化 fc 權重列張成（維度 ≤ C−1=6 / 512）；scale-free。
    隨機方向的 frac ≈ 6/512 ≈ 1.2%（baseline）。frac 越小＝位移越垂直於分類器＝越無害。
  margin = cartoon-BN 下 logit top1−top2（越大越耐位移）。

判：async frac_class < sync ⇒ 假設1（位移更 class-irrelevant）＝答案；async margin > sync ⇒ 假設2。

sanity：cartoon-BN 特徵過 fc 的 acc≈oracle~83；source-BN 的 acc≈floor(48-53)；隨機 frac≈r/512。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import util
from bn_signal_gating import build_backbone, load_backbone_only
from bn_recalib_probe import recompute_bn, node_files
from test_domain_ood_scores import load_pacs_test_data

SETTINGS = [
    ("sync",  "exp_result_v1_stage2_leave_cartoon_seed2026_topo1234"),
    ("async", "exp_result_v1_stage2_leave_cartoon_async_const_tau1e-5_style_seed2026_topo1234"),
]
SKETCH = [6, 7, 8]
DR = "../datasets/"


@torch.no_grad()
def collect(model, loader):
    model.eval()
    F, L, Y = [], [], []
    for batch in loader:
        x, y, _ = util.unpack_batch(batch)
        f = model.intermediate_forward(x.cuda(non_blocking=True))   # [B,512] fc 前、無 style shift
        lg = model.backbone.fc(f)                               # [B,C]
        F.append(f.cpu()); L.append(lg.cpu()); Y.append(y)
    return torch.cat(F), torch.cat(L), torch.cat(Y)


def class_subspace_basis(W):
    """中心化 fc 權重列張成子空間的正交基 [512, r]（r≤C-1）。"""
    Wc = W - W.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Wc, full_matrices=False)   # Vh: [C,512]
    r = int((S > 1e-6).sum())
    return Vh[:r].T, r                                       # [512, r]


def main():
    dev = "cuda"
    cart, _ = load_pacs_test_data(DR, "cartoon", 64, 4)
    sk, _ = load_pacs_test_data(DR, "sketch", 64, 4)
    agg = {}
    sanity_bad = 0
    for lab, ckpt_dir in SETTINGS:
        files = node_files(ckpt_dir, 9)
        ck0 = torch.load(files[0], map_location="cpu", weights_only=False)
        nc = ck0["backbone_state"]["backbone.fc.weight"].shape[0]
        bb = build_backbone(ck0["args"], nc, dev)
        rows = []
        print(f"\n===== {lab} =====")
        for j in SKETCH:
            # cartoon-BN 狀態
            load_backbone_only(files[j], bb, dev); recompute_bn(bb, cart, dev)
            fc, lc, y = collect(bb, cart)
            W = bb.backbone.fc.weight.detach().cpu()
            basis, r = class_subspace_basis(W)
            # source-BN 狀態
            load_backbone_only(files[j], bb, dev); recompute_bn(bb, sk, dev)
            fw, lw, _ = collect(bb, cart)
            # 幾何
            dfeat = fw - fc                                      # [N,512]
            dnorm2 = (dfeat ** 2).sum(1)
            proj2 = (dfeat @ basis) ** 2                          # [N,r]
            frac = (proj2.sum(1) / (dnorm2 + 1e-12))             # [N]
            margin = (lc.sort(1, descending=True).values[:, 0] - lc.sort(1, descending=True).values[:, 1])
            # sanity
            acc_c = (lc.argmax(1) == y).float().mean().item() * 100
            acc_w = (lw.argmax(1) == y).float().mean().item() * 100
            rnd = torch.randn_like(dfeat)
            rfrac = ((rnd @ basis) ** 2).sum(1) / (rnd ** 2).sum(1)
            ok = (75 < acc_c < 90) and (40 < acc_w < 65)
            if not ok:
                sanity_bad += 1
            print(f"  node_{j}: frac_class={frac.mean()*100:5.2f}%  margin={margin.mean():5.2f}  "
                  f"|Δfeat|={dnorm2.sqrt().mean():6.2f}  [sanity acc_c={acc_c:.1f}(~83) acc_w={acc_w:.1f}(floor) "
                  f"rand_frac={rfrac.mean()*100:.2f}%(~{100*r/512:.1f}) {'OK' if ok else '⚠BAD'}]")
            rows.append((frac.mean().item()*100, margin.mean().item(), dnorm2.sqrt().mean().item()))
        agg[lab] = np.array(rows).mean(0)

    print("\n================ 判決 (sketch 群均) ================")
    print(f"{'':8s} {'frac_class':>11s} {'margin':>8s} {'|Δfeat|':>8s}")
    for lab, _ in SETTINGS:
        a = agg[lab]
        print(f"{lab:8s} {a[0]:10.2f}% {a[1]:8.2f} {a[2]:8.2f}")
    fs, fa = agg["sync"][0], agg["async"][0]
    ms, ma = agg["sync"][1], agg["async"][1]
    print(f"\n  假設1 (位移更 class-irrelevant): async frac<sync? {fa:.2f}<{fs:.2f} → "
          f"{'成立（答案！async 位移更垂直於分類器）' if fa < fs - 0.3 else '否/不明顯'}")
    print(f"  假設2 (margin 更大): async margin>sync? {ma:.2f}>{ms:.2f} → "
          f"{'成立' if ma > ms * 1.05 else '否/不明顯'}")
    print(f"  |Δfeat| 對照(應相近、確認位移量沒差): sync={agg['sync'][2]:.1f} async={agg['async'][2]:.1f}")
    print(f"\n[sanity] 不符節點數={sanity_bad}（>0 則不可信）")


if __name__ == "__main__":
    main()
