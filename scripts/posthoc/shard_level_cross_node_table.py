"""同一批資料，換到不同節點上跑（模型與原型都用該節點自己的），差多少？

dany 2026-08-19 指定的兩類比較：
  第一類：node 0 的【訓練子集】拿去 node 1 跑（同風格、不同子集、不同節點）
  第二類：art 的資料拿去 photo 節點跑（跨風格）

★ 與先前所有實驗的差別：本腳本用【逐節點的訓練子集】，不是整個域。
   PACS 沒有 train/test 目錄分割（`pacs_dataset.py`：ImageFolder 讀整個域資料夾），
   訓練時的 test_loader 只載入 leave-out 域 ⇒ 來源域「沒有」保留測試集。
   ⇒ 先前的 ① 其實是三個節點訓練子集的聯集，本腳本把它拆開。

分割重建：`util.partition_domain_dataset_for_nodes`，split_mode=class_balanced（graphid 6 的預設）、
seed=2026，person 在分割【之後】才排除（與 util.py L825-831 同序）。
自檢＝逐節點樣本數必須對上訓練 log：535/532/532 | 415/412/411 | 1257/1257/1254。
"""
import os, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from pacs_dataset import PACSDataset
from util import partition_domain_dataset_for_nodes
from osdg_eval import load_backbone_diffusion
from dood.prototype import class_centers

DESC = "v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
CK = "exp_result_" + DESC
ROOT = "../datasets/"; SEED = 2026; UNK = 6; DEG = 57.29577951308232
SRC = ["art_painting", "photo", "sketch"]; LEAVE = "cartoon"
OWN = {0: "art_painting", 1: "art_painting", 2: "art_painting",
       3: "photo", 4: "photo", 5: "photo",
       6: "sketch", 7: "sketch", 8: "sketch"}
SHORT = {"art_painting": "art", "photo": "photo", "sketch": "sketch", "cartoon": "cartoon"}

TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# ── 重建每個節點的訓練子集 ──
GROUPS = {}      # 名稱 -> DataLoader
COUNTS = {}
for d in SRC:
    ds = PACSDataset(root=ROOT, dataset_name=d, transform=TF)
    names = [f"node_{i}" for i in range(9) if OWN[i] == d]
    part = partition_domain_dataset_for_nodes(ds, names, split_mode="class_balanced", seed=SEED)
    for nm in names:
        idxs = [i for i in part[nm] if ds.targets[i] != UNK]      # person 在分割後才排除
        GROUPS[f"{nm}的{SHORT[d]}訓練子集"] = DataLoader(Subset(ds, idxs), batch_size=64,
                                                       shuffle=False, num_workers=4)
        COUNTS[f"{nm}的{SHORT[d]}訓練子集"] = len(idxs)
_c = PACSDataset(root=ROOT, dataset_name=LEAVE, transform=TF)
_ci = [i for i in range(len(_c)) if _c.targets[i] != UNK]
GROUPS["cartoon(無人訓練過)"] = DataLoader(Subset(_c, _ci), batch_size=64, shuffle=False, num_workers=4)
COUNTS["cartoon(無人訓練過)"] = len(_ci)
GNAMES = list(GROUPS)

print("=" * 84)
print("§0 自檢：重建的訓練子集樣本數 vs 訓練 log")
print("=" * 84)
EXP = {"node_0": 535, "node_1": 532, "node_2": 532, "node_3": 415, "node_4": 412,
       "node_5": 411, "node_6": 1257, "node_7": 1257, "node_8": 1254}
ok = True
for g in GNAMES:
    if "訓練子集" not in g: continue
    nm = g.split("的")[0]
    good = COUNTS[g] == EXP[nm]
    ok &= good
    print(f"  {g:<28} n={COUNTS[g]:<6} log={EXP[nm]:<6} {'✅' if good else '❌'}")
print(f"  {'cartoon(無人訓練過)':<28} n={COUNTS['cartoon(無人訓練過)']}（已排除 person）")
print(f"  ⇒ 分割重建 {'完全一致' if ok else '★不一致，以下數字不可用★'}")
assert ok, "訓練子集重建與 log 不符，中止"


@torch.no_grad()
def angle(bb, C, loader):
    """該批資料到【自己那一類】原型中心的平均角度（度）"""
    out = []
    for x, y, _ in loader:
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x.to("cuda")))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        _, v = bb.forward_from_layer3(h)
        z = bb.project(v).cpu().numpy()
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        yy = np.asarray(y).flatten().astype(int)
        out.append(np.arccos(np.clip((z @ C.T)[np.arange(len(yy)), yy], -1 + 1e-7, 1 - 1e-7)) * DEG)
    return float(np.concatenate(out).mean())


A = np.zeros((len(GNAMES), 9))          # [資料組][跑在哪個節點]
for j in range(9):
    bb, _ = load_backbone_diffusion(os.path.join(CK, f"{DESC}_node_{j}_final.pth"), 6, "cuda")
    C = class_centers(bb.prototypes, bb.proto_count).cpu().numpy()
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    for gi, g in enumerate(GNAMES):
        A[gi, j] = angle(bb, C, GROUPS[g])
    del bb
    print(f"  node_{j} ({SHORT[OWN[j]]}) 跑完", flush=True)

np.save(os.path.join(os.path.dirname(__file__), "..", "..", "logs", "prototype_probe",
                     "0819_shard_cross_node.npy"), A)

W = 96
print("\n" + "=" * W)
print("【主表】同一批資料 × 跑在哪個節點（模型與原型都是該節點自己的），單位＝度")
print("=" * W)
print(f"{'資料':<26}{'n':>6}" + "".join(f"{'nd'+str(j):>7}" for j in range(9)))
print(f"{'':<26}{'':>6}" + "".join(f"{SHORT[OWN[j]][:5]:>7}" for j in range(9)))
print("-" * W)
for gi, g in enumerate(GNAMES):
    print(f"{g:<24}{COUNTS[g]:>7}" + "".join(f"{A[gi,j]:>7.2f}" for j in range(9)))

print("\n" + "=" * W)
print("★ 第一類：同風格、不同節點（以 art 三個節點為例；其他風格同理）")
print("=" * W)
for d in SRC:
    ids = [i for i in range(9) if OWN[i] == d]
    self_ = np.mean([A[GNAMES.index(f"node_{i}的{SHORT[d]}訓練子集"), i] for i in ids])
    cross = np.mean([A[GNAMES.index(f"node_{i}的{SHORT[d]}訓練子集"), j] for i in ids for j in ids if j != i])
    print(f"  {SHORT[d]:<8} 資料在【自己節點】 {self_:>7.2f}°   在【同風格別的節點】 {cross:>7.2f}°"
          f"   差 {cross-self_:+6.2f}°")
allself = np.mean([A[GNAMES.index(f"node_{i}的{SHORT[OWN[i]]}訓練子集"), i] for i in range(9)])
allcross = np.mean([A[GNAMES.index(f"node_{i}的{SHORT[OWN[i]]}訓練子集"), j]
                    for i in range(9) for j in range(9) if j != i and OWN[j] == OWN[i]])
print(f"  {'全部':<8} {allself:>25.2f}°{allcross:>24.2f}°   差 {allcross-allself:+6.2f}°")

print("\n" + "=" * W)
print("★ 第二類：跨風格（資料的風格 vs 跑在哪個風格的節點）")
print("=" * W)
print(f"{'資料的風格':<12}" + "".join(f"{'跑在'+SHORT[d]+'節點':>16}" for d in SRC))
for d in SRC:
    row = f"{SHORT[d]:<14}"
    for d2 in SRC:
        js = [j for j in range(9) if OWN[j] == d2]
        gs = [GNAMES.index(f"node_{i}的{SHORT[d]}訓練子集") for i in range(9) if OWN[i] == d]
        v = np.mean([A[g, j] for g in gs for j in js])
        row += f"{v:>14.2f}°" + ("  ←自己" if d == d2 else "      ")
    print(row)
gc = GNAMES.index("cartoon(無人訓練過)")
print(f"{'cartoon':<14}" + "".join(
    f"{np.mean([A[gc,j] for j in range(9) if OWN[j]==d2]):>14.2f}°      " for d2 in SRC))
print(f"\n  （cartoon 沒有主人，三欄都是「別人的節點」）")
