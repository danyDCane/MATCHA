"""換機重現驗收（1028）：新 run 與登記 run 逐位比對。全部相同 exit 0，否則 exit 1。

  --ckpts epoch_5            比這些存檔點的九台 checkpoint（每個張量 torch.equal）
  --diag broadcast_log.csv   比 async_diag 下的 CSV；--prefix 時只比新檔的共同前綴（最後一列可能寫一半，不比）
  --eval REF.csv NEW.csv     比 osdg_eval 輸出（ckpt×node×score_fn 全欄）
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
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
