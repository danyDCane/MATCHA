"""換機重現驗收（1028）：新 run 與登記 run 逐位比對。全部相同 exit 0，否則 exit 1。

  --ckpts epoch_5            比這些存檔點的九台 checkpoint（每個張量 torch.equal）
  --diag broadcast_log.csv   比 async_diag 下的 CSV；--prefix 時只比新檔的共同前綴（最後一列可能寫一半，不比）
  --eval REF.csv NEW.csv     比 osdg_eval 輸出（ckpt×node×score_fn 全欄）
  [1103 新增，舊用法不受影響]
  --args_ckpt epoch_50       比 node_0 該存檔點裡的 args 全欄位；--allow_diff 列可不同的鍵（預設只放名稱類）；
                             --allow_new_keys 允許新 run 多出鍵（逐一印出值）——閘門用；折間比對不加
  --log REF.log NEW.log      比訓練 log 的數字行（到 --log_upto 那一個 epoch 行之前），新名稱換回登記名稱再比
  --assert_sync NEW.log      斷言新 run 確實是同步：log 無 [ASYNC 行、拓樸 edges=--edges、硬體守門可逐位重現、
                             run 目錄無 async_diag/、args 的非同步開關皆關；加 --div_ckpt epoch_200 再查九台參數分歧 < --div_max
"""
import argparse
import csv
import sys

import torch


def flat(o, p=""):
    r = {}
    if torch.is_tensor(o):
        r[p] = o
    elif isinstance(o, dict):
        for k, v in o.items():
            r.update(flat(v, f"{p}.{k}" if p else str(k)))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            r.update(flat(v, f"{p}[{i}]"))
    return r


ap = argparse.ArgumentParser()
ap.add_argument("--ref_dir", required=True)
ap.add_argument("--new_dir", required=True)
ap.add_argument("--ref_name", required=True, help="登記 run 的 checkpoint 檔名前綴")
ap.add_argument("--new_name", default=None, help="新 run 的檔名前綴（預設同 ref_name）")
ap.add_argument("--ckpts", nargs="*", default=[])
ap.add_argument("--diag", nargs="*", default=[])
ap.add_argument("--prefix", action="store_true")
ap.add_argument("--eval", nargs=2, default=None)
ap.add_argument("--args_ckpt", default=None)
ap.add_argument("--allow_diff", nargs="*", default=["name", "description", "savePath", "wandb_project"])
ap.add_argument("--allow_new_keys", action="store_true")
ap.add_argument("--log", nargs=2, default=None)
ap.add_argument("--log_upto", type=int, default=51)
ap.add_argument("--assert_sync", default=None)
ap.add_argument("--edges", type=int, default=23)
ap.add_argument("--div_ckpt", default=None, help="assert_sync 附加：該存檔點九台參數分歧需 < --div_max")
ap.add_argument("--div_max", type=float, default=1e-6)
a = ap.parse_args()
new_name = a.new_name or a.ref_name
ok = True
for ep in a.ckpts:
    tot = eq = 0
    bad = []
    for n in range(9):
        A = flat(torch.load(f"{a.ref_dir}/{a.ref_name}_node_{n}_{ep}.pth", map_location="cpu", weights_only=False))
        B = flat(torch.load(f"{a.new_dir}/{new_name}_node_{n}_{ep}.pth", map_location="cpu", weights_only=False))
        if A.keys() != B.keys():
            bad.append((n, "key mismatch"))
        for k in A:
            tot += 1
            if k in B and torch.equal(A[k], B[k]):
                eq += 1
            elif len(bad) < 4:
                bad.append((n, k))
    print(f"[ckpt {ep}] 張量 {eq}/{tot} 逐位相同 {bad if bad else ''}")
    ok &= eq == tot
for f in a.diag:
    A = list(csv.reader(open(f"{a.ref_dir}/async_diag/{f}")))
    B = list(csv.reader(open(f"{a.new_dir}/async_diag/{f}")))
    m = len(B) - 1 if a.prefix else max(len(A), len(B))
    first = next((i for i in range(m) if i >= len(A) or i >= len(B) or A[i] != B[i]), None)
    same = first is None and (a.prefix or len(A) == len(B))
    print(f"[diag {f}] 登記 {len(A)} 列／新 {len(B)} 列，比 {m} 列："
          + ("全部逐字相同" if same else f"第 {first} 列不同 REG={A[first][:5] if first is not None and first < len(A) else '-'} "
                                         f"NEW={B[first][:5] if first is not None and first < len(B) else '-'}"))
    ok &= same
if a.eval:
    M = ["oscr", "h_best", "h_accK", "h_accU", "det_auroc", "det_aupr", "fpr95", "closed_acc"]
    ka = {(r["ckpt"], r["node"], r["score_fn"]): [r[m] for m in M] for r in csv.DictReader(open(a.eval[0]))}
    kb = {(r["ckpt"], r["node"], r["score_fn"]): [r[m] for m in M] for r in csv.DictReader(open(a.eval[1]))}
    nd = sum(ka[k] != kb.get(k) for k in ka) + len(set(kb) - set(ka))
    print(f"[eval] 登記 {len(ka)} 列／新 {len(kb)} 列：" + ("全部逐字相同" if nd == 0 else f"{nd} 列不同"))
    ok &= nd == 0

def load_args(d, name, ep):
    ck = torch.load(f"{d}/{name}_node_0_{ep}.pth", map_location="cpu", weights_only=False)
    x = ck.get("args")
    return dict(x) if isinstance(x, dict) else vars(x)


if a.args_ckpt:
    A = load_args(a.ref_dir, a.ref_name, a.args_ckpt)
    B = load_args(a.new_dir, new_name, a.args_ckpt)
    only_ref = sorted(set(A) - set(B))
    only_new = sorted(set(B) - set(A))
    diff = sorted(k for k in set(A) & set(B) if A[k] != B[k] and k not in a.allow_diff)
    allowed = sorted(k for k in set(A) & set(B) if A[k] != B[k] and k in a.allow_diff)
    print(f"[args {a.args_ckpt}] 登記 {len(A)} 鍵／新 {len(B)} 鍵；容許不同且確實不同：{allowed}")
    for k in allowed:
        print(f"    容許  {k}: {A[k]!r} → {B[k]!r}")
    for k in only_new:
        print(f"    新增鍵 {k} = {B[k]!r}")
    for k in only_ref:
        print(f"    ❌ 新 run 缺鍵 {k}（登記值 {A[k]!r}）")
    for k in diff:
        print(f"    ❌ 不容許的差異 {k}: {A[k]!r} → {B[k]!r}")
    good = not diff and not only_ref and (a.allow_new_keys or not only_new)
    if only_new and not a.allow_new_keys:
        print("    ❌ 新 run 多出鍵但未給 --allow_new_keys")
    print(f"[args] {'全欄位符合' if good else '不符'}")
    ok &= good

if a.log:
    PFX = ("Epoch ", "[confidence]", "[BN-DIV]", "⚠ WARNING: Parameters", "  [proto]", "         comp:",
           "[TOPO]", "[proto] stage", "[Checkpoint]", "[OSDG]", "[PACS", "[Single Process]")
    stop = f"Epoch {a.log_upto}:"

    def grab(path, rename=None):
        keep, other = [], {}
        for ln in open(path, encoding="utf-8", errors="replace"):
            ln = ln.rstrip("\n")
            if rename:
                ln = ln.replace(rename[0], rename[1])
            if ln.startswith(stop):
                return keep, other, True
            if ln.startswith(PFX):
                keep.append(ln)
            elif ln.strip() and not ln.startswith(("wandb:", "Authorization", "==>", "=====")):
                other[ln[:40]] = other.get(ln[:40], 0) + 1
        return keep, other, False

    LA, OA, fa = grab(a.log[0])
    LB, OB, fb = grab(a.log[1], rename=(new_name, a.ref_name) if new_name != a.ref_name else None)
    first = next((i for i in range(max(len(LA), len(LB)))
                  if i >= len(LA) or i >= len(LB) or LA[i] != LB[i]), None)
    same = first is None and fa and fb
    print(f"[log] 到 '{stop}' 之前：登記 {len(LA)} 行／新 {len(LB)} 行（兩邊都讀到該行：{fa}/{fb}）："
          + ("全部逐字相同" if same else
             f"第 {first} 行不同\n    REG={LA[first] if first is not None and first < len(LA) else '-'}"
             f"\n    NEW={LB[first] if first is not None and first < len(LB) else '-'}"))
    extra = {k: v for k, v in OB.items() if k not in OA}
    if extra:
        print(f"    新 log 有、登記 log 沒有的其他行（不參與比對，供人工檢查）：{extra}")
    ok &= bool(same)

if a.assert_sync:
    import os
    txt = open(a.assert_sync, encoding="utf-8", errors="replace").read().splitlines()
    n_async = sum(1 for ln in txt if ln.startswith("[ASYNC"))
    n_cons = sum(1 for ln in txt if ln.startswith("⚠ WARNING: Parameters INCONSISTENT after comm iter"))
    topo = [ln for ln in txt if ln.startswith("[TOPO]")]
    hw = [ln for ln in txt if ln.startswith("[hw_guard]")]
    has_diag = os.path.isdir(f"{a.new_dir}/async_diag")
    ep = a.args_ckpt or "epoch_50"
    B = load_args(a.new_dir, new_name, ep)
    want = {"async_trigger": False, "async_style": False, "aggregate_bn": True, "use_proto_reg": True,
            "isolated_nodes": False, "graphid": 6}
    bad_args = {k: B.get(k, "<缺>") for k, v in want.items() if B.get(k, "<缺>") != v}
    # ⚠️ 「參數一致性」警告行非同步也會印（train.py 每 epoch 比第一個參數到小數 4 位）⇒ 只印參考、不當判準
    print(f"[sync] （參考）參數一致性警告行 {n_cons} 行——同步／非同步都會印，不能證明是同步")
    checks = [
        ("log 無 [ASYNC 行", n_async == 0, f"{n_async} 行"),
        (f"拓樸 edges={a.edges}", len(topo) == 1 and f"edges={a.edges}" in topo[0], topo[0] if topo else "無 [TOPO] 行"),
        ("硬體守門＝可逐位重現", len(hw) == 1 and "可逐位重現" in hw[0], hw[0][:120] if hw else "無 [hw_guard] 行"),
        ("run 目錄無 async_diag/", not has_diag, "有" if has_diag else "無"),
        (f"args 非同步開關皆關（{ep}）", not bad_args, str(bad_args) if bad_args else str(want)),
    ]
    if a.div_ckpt:
        # 同步每步全體 MH 平均 ⇒ 訓練末九台可訓練參數幾乎相同。1103 實測 epoch_200 相對 L2 分歧：
        # 同步 9.20e-08（A2 cartoon）／9.36e-08（StyleDDG sketch）vs 非同步 4.57e-05／1.50e-04（A0 cartoon／sketch）
        sds = [torch.load(f"{a.new_dir}/{new_name}_node_{i}_{a.div_ckpt}.pth", map_location="cpu",
                          weights_only=False)["backbone_state"] for i in range(9)]
        ks = [k for k, v in sds[0].items() if torch.is_tensor(v) and v.is_floating_point()
              and not any(t in k for t in ("running_", "num_batches", "prototypes", "proto_count", "diffusion"))]
        num = den = 0.0
        for k in ks:
            X = torch.stack([s[k].double() for s in sds]); m = X.mean(0)
            num += ((X - m) ** 2).sum().item(); den += (9 * (m ** 2).sum()).item()
        r = (num / den) ** 0.5
        checks.append((f"九台可訓練參數相對 L2 分歧（{a.div_ckpt}）< {a.div_max:g}", r < a.div_max,
                       f"{r:.2e}（{len(ks)} 個張量）"))
    for nm, g, info in checks:
        print(f"[sync] {'✅' if g else '❌'} {nm}：{info}")
    ok &= all(g for _, g, _ in checks)

print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
