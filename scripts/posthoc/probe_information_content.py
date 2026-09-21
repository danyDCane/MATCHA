"""四格（實為六格）探針：投影層是「毀了資訊」還是「藏了資訊」？（dany 2026-08-26 設計）

比喻對齊：512 維 h ＝ 一整箱沒整理的工具（東西都在，要翻）；
          128 維 Z ＝ 掛在牆上的工具板（一件沒多，但伸手就拿得到）。
          「六個距離」這個讀出極度簡單 ⇒ 投影層的工作是「排」，讓簡單讀出撈得到。

三個對照各答一個問題：
  MLP(Z) vs MLP(h)   ⇒ 投影層有沒有毀掉資訊（漏斗漏了多少）
  線性(Z) vs MLP(Z)  ⇒ Z 裡的資訊排好了沒（工具板整不整齊）
  線性(Z) vs 0.8264  ⇒ 「128維→六個距離」漏掉多少（讀出的浪費）

判讀（事前寫死，dany）：
  MLP(Z) 明顯＜MLP(h)            ⇒ 漏斗真的漏 ⇒ 必須重新設計投影層；換讀出全部封頂在 MLP(Z)，不要投資
  MLP(Z)≈MLP(h) 但線性(Z) 低      ⇒ 東西在、排得亂 ⇒ 改投影層的損失，不用推翻架構
  線性(Z) 高（0.88+）             ⇒ 東西在也排好了，是尺撈不出來 ⇒ 投影層不用動，全力換讀出

⚠️ 三個誠實界線：
  ① 5 折交叉驗證、只報 held-out（②1939 vs ③405，MLP 參數量遠超樣本數，全體訓練＝背答案）
  ② h 跑「原始」與「L2 正規化」兩版——Z 已被正規化丟掉範數，不對齊就分不清「毀了」還是「只是少了範數」
     順帶：h原始 − h正規化 ＝ 範數本身值多少
  ③ 探針用 ②③ 的標籤在 cartoon 上訓練＝「已看過目標畫風的答案」⇒ 回答「資訊在不在」，不是「方法拿不拿得到」

同時落盤：512維 h、128維 Z、範數、layer1/2/3 通道統計、logits ⇒ 後續候選不必再 forward。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch, util
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers
from bn_common import bn_avg, apply_bn

PACS = ["art_painting", "cartoon", "photo", "sketch"]
DESC = os.environ.get("DESC", "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix")
CKPT = os.environ.get("CKPT_TAG", "final"); NODES = int(os.environ.get("NODES", "9"))
DUMP = os.environ.get("DUMP", "logs/prototype_probe/0826_features_full.npz")
CK = "exp_result_" + DESC
leave = "cartoon"; UNK = 6; N = 9; NC = 6
avail = [d for d in PACS if d != leave]; per = N // len(avail)
OWN = [avail[min(i // per, len(avail) - 1)] for i in range(N)]
nrm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)


@torch.no_grad()
def collect(bb, loader):
    """回傳 dict：h512(未正規化)、z128(已正規化)、logit、y、以及 layer1/2/3 的通道 mean/std"""
    out = {k: [] for k in "h z lo y s1 s2 s3".split()}
    for b in loader:
        x, y, _ = util.unpack_batch(b)
        t = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        f1 = bb.backbone.layer1(t); f2 = bb.backbone.layer2(f1); f3 = bb.backbone.layer3(f2)
        lo, v = bb.forward_from_layer3(f3)
        for tag, f in [("s1", f1), ("s2", f2), ("s3", f3)]:
            out[tag].append(torch.cat([f.mean((2, 3)), f.std((2, 3))], 1).cpu().numpy())
        out["h"].append(v.cpu().numpy()); out["z"].append(bb.project(v).cpu().numpy())
        out["lo"].append(lo.cpu().numpy()); out["y"].append(np.asarray(y).flatten())
    return {k: np.concatenate(v) for k, v in out.items()}


if not os.path.exists(DUMP):
    AVG = bn_avg(CK, DESC, N=N, ckpt_tag=CKPT)
    D = {}
    for i in range(NODES):
        bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{i}_{CKPT}.pth"), 6, "cuda")
        apply_bn(bb, AVG)
        C = nrm(class_centers(bb.prototypes, bb.proto_count).cpu().numpy())
        ld = {d: TD.load_pacs_test_data("../datasets/", d, 64, 4)[0] for d in [OWN[i], leave]}
        t = collect(bb, ld[leave]); s = collect(bb, ld[OWN[i]])
        del bb
        for k, v in t.items(): D[f"n{i}_tgt_{k}"] = v.astype(np.float16 if k != "y" else np.int16)
        for k, v in s.items(): D[f"n{i}_src_{k}"] = v.astype(np.float16 if k != "y" else np.int16)
        D[f"n{i}_C"] = C.astype(np.float32)
        print(f"  forward node{i}({OWN[i]}) done", flush=True)
    os.makedirs(os.path.dirname(DUMP), exist_ok=True)
    np.savez_compressed(DUMP, **D)
    print(f"落盤 → {DUMP}  ({os.path.getsize(DUMP)/1e6:.0f} MB)", flush=True)

F = np.load(DUMP)
SEED = 2026
GRIDS = [("512維 h（原始）", "h_raw"), ("512維 h（L2正規化）", "h_nrm"), ("128維 Z", "z")]


def probe(X, y, kind):
    """5 折交叉驗證、只回傳 held-out AUROC"""
    sk = StratifiedKFold(5, shuffle=True, random_state=SEED)
    sc = []
    for tr, te in sk.split(X, y):
        if kind == "lin":
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        else:
            m = make_pipeline(StandardScaler(),
                              MLPClassifier((256,), max_iter=400, random_state=SEED,
                                            early_stopping=True, n_iter_no_change=15))
        m.fit(X[tr], y[tr])
        sc.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(sc))


R = {}
for i in range(NODES):
    y = F[f"n{i}_tgt_y"].astype(int)
    h = F[f"n{i}_tgt_h"].astype(np.float32); z = F[f"n{i}_tgt_z"].astype(np.float32)
    lab = (y == UNK).astype(int)
    X = {"h_raw": h, "h_nrm": nrm(h), "z": nrm(z)}
    for gname, gk in GRIDS:
        for kind in ["lin", "mlp"]:
            R.setdefault((gname, kind), []).append(probe(X[gk], lab, kind))
    R.setdefault("norm_only", []).append(probe(np.linalg.norm(h, axis=1, keepdims=True), lab, "lin"))
    print(f"  probe node{i} done", flush=True)

m = lambda k: float(np.mean(R[k]))
print("\n" + "=" * 92)
print("★ 資訊含量四格（5 折交叉驗證、held-out AUROC、9 節點平均、② vs ③）")
print(f"{'空間':<22}{'線性探針':>12}{'MLP 探針':>12}{'MLP−線性':>12}")
print("-" * 92)
for gname, _ in GRIDS:
    l, mm = m((gname, "lin")), m((gname, "mlp"))
    print(f"{gname:<22}{l:>12.4f}{mm:>12.4f}{mm-l:>+12.4f}")
print("-" * 92)
print(f"{'（僅 512 維範數 1 維）':<22}{m('norm_only'):>12.4f}")
print("\n★ 三個對照")
print(f"  ① MLP(Z) {m(('128維 Z','mlp')):.4f}  vs  MLP(h原始) {m(('512維 h（原始）','mlp')):.4f} "
      f" ⇒ 差 {m(('128維 Z','mlp'))-m(('512維 h（原始）','mlp')):+.4f}")
print(f"     公平版：MLP(Z) vs MLP(h正規化) {m(('512維 h（L2正規化）','mlp')):.4f}"
      f" ⇒ 差 {m(('128維 Z','mlp'))-m(('512維 h（L2正規化）','mlp')):+.4f}   ← 投影層有沒有毀掉資訊")
print(f"  ② 線性(Z) {m(('128維 Z','lin')):.4f}  vs  MLP(Z) {m(('128維 Z','mlp')):.4f}"
      f" ⇒ 差 {m(('128維 Z','mlp'))-m(('128維 Z','lin')):+.4f}   ← Z 裡排好了沒")
print(f"  ③ 線性(Z) {m(('128維 Z','lin')):.4f}  vs  六距離飽和 0.8264"
      f" ⇒ 差 {m(('128維 Z','lin'))-0.8264:+.4f}   ← 讀出的浪費")
print(f"  ★ 範數本身值多少：MLP(h原始) − MLP(h正規化) = "
      f"{m(('512維 h（原始）','mlp'))-m(('512維 h（L2正規化）','mlp')):+.4f}")
print(f"\n錨點：現行讀出 0.8145 ｜ 六距離飽和 0.8264 ｜ energy 0.8380 ｜ −std(logit) 0.8656")
print("=" * 92)
