#!/usr/bin/env python
"""來源側訊號天花板：四個量測（2026-09-16／09-22，獨立管線複核用）

回答一個問題：**來源側拿得到的訊號，看不看得見通往目標畫風的那個方向？**

四個子命令（互相獨立，可單獨跑）：
  merge     跨節點原型「合併 vs 平均」當檢測參照物
  styleaxis 沿畫風軸壓縮：oracle（含目標資料）vs 來源側（零目標資訊）vs 隨機零點
  capacity  縮小有效容量（限制到來源側前 k 個主方向）對檢測與散布的影響
  thresh    門檻誤差拆解：節點間離散 vs 來源→目標偏移

資料：logs/prototype_probe/0908_dump/full_{sketch,cartoon,art_painting,photo}.npz
口徑：面讀出 ‖z − QQᵀz‖、參照物 Cema、full 臂、ep200 final、9 節點平均、person=類別 6。
⚠️ 全部是 post-hoc 幾何量測，**不是訓練**——引用時射程必帶。

用法：venv_matcha/bin/python scripts/posthoc/source_side_signal_ceiling.py merge
"""
import sys, os, argparse
import numpy as np

DUMP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "logs/prototype_probe/0908_dump")
FOLDS = ["sketch", "cartoon", "art_painting", "photo"]
UNK = 6          # PACS 類別順序 dog/elephant/giraffe/guitar/horse/house/person
KNOWN = list(range(6))


def load(arm, fold):
    return np.load(f"{DUMP}/{arm}_{fold}.npz")


def auroc(s, y):
    """y=1 為未知。自寫，不用 sklearn（獨立管線）。平手取平均秩。"""
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[order] = np.arange(1, len(s) + 1)
    su = s[order]
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def nrm(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def face(V, C):
    """面讀出：到原型張成子空間的殘差長度。"""
    Q, _ = np.linalg.qr(C.T)
    return np.linalg.norm(V - (V @ Q) @ Q.T, axis=1)


def width(s, u):
    """本專案 §3b 口徑的寬度：兩堆標準差的均方根。"""
    return np.sqrt((s[u].std(ddof=1) ** 2 + s[~u].std(ddof=1) ** 2) / 2)


# ─────────────────────────────────────────────────────────────
def cmd_merge(_):
    print("跨節點原型當檢測參照物：合併 vs 平均（面讀出、目標域、9 節點平均）")
    print(f'{"fold":13s} {"現行·自己6":>10s} {"合併18":>8s} {"合併54":>8s} {"平均6":>8s}')
    acc = {k: [] for k in ("own", "m18", "m54", "avg")}
    for f in FOLDS:
        d = load("full", f); r = {k: [] for k in acc}
        allC = np.stack([d[f"n{n}_Cema"] for n in range(9)])          # 9×6×128
        m54 = allC.reshape(-1, 128)
        avg = nrm(allC.mean(0))
        for n in range(9):
            V = d[f"n{n}_tgt_z"]; y = (d[f"n{n}_tgt_y"] == UNK).astype(int)
            m18 = np.concatenate([d[f"n{3 * g + (n % 3)}_Cema"] for g in range(3)])
            for k, C in (("own", d[f"n{n}_Cema"]), ("m18", m18), ("m54", m54), ("avg", avg)):
                r[k].append(auroc(face(V, C), y))
        for k in acc:
            acc[k].append(np.mean(r[k]))
        print(f'{f:13s} ' + ' '.join(f'{np.mean(r[k]):10.4f}' if k == "own"
                                     else f'{np.mean(r[k]):8.4f}' for k in acc))
    print(f'{"四折平均":13s} ' + ' '.join(f'{np.mean(v):10.4f}' if k == "own"
                                          else f'{np.mean(v):8.4f}' for k, v in acc.items()))


def cmd_styleaxis(a):
    rng = np.random.default_rng(a.seed)
    print("沿畫風軸壓縮（λ=1 完全移除該子空間）")
    print(f'{"fold":13s} {"基準":>7s} | {"來源軸12維":>10s} {"隨機12維":>9s} {"Δ":>8s} | '
          f'{"oracle6維":>9s} {"隨機6維":>8s} {"Δ":>8s}')

    def run(d, getQ):
        A = []
        for n in range(9):
            tz = d[f"n{n}_tgt_z"]; ty = d[f"n{n}_tgt_y"]; C = nrm(d[f"n{n}_Cema"])
            Q = getQ(d, n)
            g = lambda X: nrm(X - (X @ Q) @ Q.T) if Q.shape[1] else nrm(X)
            A.append(auroc(face(g(tz), nrm(g(C))), (ty == UNK).astype(int)))
        return float(np.mean(A))

    def Q_src(d, n):           # 只用跨節點來源側原型 ⇒ 零目標資訊
        P = np.stack([d[f"n{3 * g + (n % 3)}_Cema"] for g in range(3)])
        Q, _ = np.linalg.qr((P - P.mean(0)).reshape(-1, 128).T)
        return Q[:, :12]

    def Q_orc(d, n):           # 來源→目標的畫風方向 ⇒ 需要目標資料（oracle）
        sz, sy = d[f"n{n}_src_z"], d[f"n{n}_src_y"]
        tz, ty = d[f"n{n}_tgt_z"], d[f"n{n}_tgt_y"]
        D = np.stack([tz[ty == c].mean(0) - sz[sy == c].mean(0) for c in KNOWN])
        Q, _ = np.linalg.qr(D.T)
        return Q[:, :6]

    def Q_rand(k):
        def g(d, n):
            Q, _ = np.linalg.qr(rng.standard_normal((128, k)))
            return Q[:, :k]
        return g

    for f in FOLDS:
        d = load("full", f)
        base = run(d, lambda d, n: np.zeros((128, 0)))
        s12, o6 = run(d, Q_src), run(d, Q_orc)
        r12 = np.mean([run(d, Q_rand(12)) for _ in range(a.reps)])
        r6 = np.mean([run(d, Q_rand(6)) for _ in range(a.reps)])
        print(f'{f:13s} {base:7.4f} | {s12:10.4f} {r12:9.4f} {s12 - r12:+8.4f} | '
              f'{o6:9.4f} {r6:8.4f} {o6 - r6:+8.4f}')


def cmd_capacity(a):
    print("縮小有效容量：把 z 限制到【來源側】前 k 個主方向（零目標資訊）")
    ks = a.ks
    print(f'{"fold":13s} ' + ' '.join(f'k={k:<6d}' for k in ks) + '| 散布(寬度) ' +
          ' '.join(f'k={k}' for k in ks))
    for f in FOLDS:
        d = load("full", f); A = []; W = []
        for k in ks:
            aa, ww = [], []
            for n in range(9):
                sz = d[f"n{n}_src_z"]; tz = d[f"n{n}_tgt_z"]
                ty = d[f"n{n}_tgt_y"]; C = d[f"n{n}_Cema"]
                Vt = np.linalg.svd(sz - sz.mean(0), full_matrices=False)[2]
                P = Vt[:k].T
                s = face(nrm((tz @ P) @ P.T), nrm((C @ P) @ P.T))
                u = ty == UNK
                aa.append(auroc(s, u.astype(int))); ww.append(width(s, u))
            A.append(np.mean(aa)); W.append(np.mean(ww))
        print(f'{f:13s} ' + ' '.join(f'{x:.4f}  ' for x in A) + '|            ' +
              ' '.join(f'{x:.4f}' for x in W))


def cmd_thresh(_):
    print("門檻誤差拆解：來源已知的 95 分位當門檻（名目 5% 誤拒），搬到目標域")
    print(f'{"fold":13s} {"實際誤拒":>8s} {"節點間全距":>10s} | {"共用門檻後":>10s} {"全距":>7s}')
    for f in FOLDS:
        d = load("full", f); fpr = []; taus = []; S = []
        for n in range(9):
            C = d[f"n{n}_Cema"]
            ss = face(d[f"n{n}_src_z"], C); sy = d[f"n{n}_src_y"]
            st = face(d[f"n{n}_tgt_z"], C); ty = d[f"n{n}_tgt_y"]
            tau = np.percentile(ss[sy != UNK], 95)
            fpr.append((st[ty != UNK] > tau).mean()); taus.append(tau)
            S.append((st, ty))
        fpr = np.array(fpr); tb = np.mean(taus)
        fb = np.array([(st[ty != UNK] > tb).mean() for st, ty in S])
        print(f'{f:13s} {100*fpr.mean():7.1f}% {100*(fpr.max()-fpr.min()):9.1f}pp | '
              f'{100*fb.mean():9.1f}% {100*(fb.max()-fb.min()):6.1f}pp')


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("merge").set_defaults(fn=cmd_merge)
    q = sub.add_parser("styleaxis"); q.add_argument("--reps", type=int, default=5)
    q.add_argument("--seed", type=int, default=2026); q.set_defaults(fn=cmd_styleaxis)
    q = sub.add_parser("capacity"); q.add_argument("--ks", type=int, nargs="+",
                                                   default=[6, 12, 24, 48, 128])
    q.set_defaults(fn=cmd_capacity)
    sub.add_parser("thresh").set_defaults(fn=cmd_thresh)
    a = p.parse_args(); a.fn(a)
