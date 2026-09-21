"""原型讀出失效診斷：②③ 分離度 ＋ 投影前(512) vs 投影後(128) 的單一變因對照。

問題：sketch fold 的原型讀出部署 AUROC 僅 0.68，同模型 energy 有 0.85。
兩個假設：(a) ②(target 已知類) 被推遠、與 ③(person) 重疊；(b) 病灶在投影層。

設計：**同一種讀出形式（到類別中心的最小角距離），只差在 512 維或 128 維**
  - 類別中心一律**從該節點自己的源域樣本現算**（兩邊同法 ⇒ 單一變因＝投影層）
  - ⚠️ 不用 checkpoint 的 prototypes buffer 當 128 維中心：那是 EMA 累積的、與 512 維不同源，
    會引入第二個變因。本腳本另外報 buffer 版供對照。
BN 一律先跨節點平均（B 法：mean(var+mean²)−mean(mean)²），與 osdg_eval --avg_bn 一致。
"""
import os, sys, argparse, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import util, test_domain_ood_scores as TD
from osdg_eval import load_backbone_diffusion, compute_oscr
from sklearn.metrics import roc_auc_score
PACS = ["art_painting", "cartoon", "photo", "sketch"]
DEG = 57.29577951308232


def bn_avg_B(states):
    """B 法合併變異數（0819b §0.2）。"""
    keys = [k for k in states[0] if k.endswith(("running_mean", "running_var"))]
    out = {}
    for k in keys:
        if k.endswith("running_mean"):
            out[k] = torch.stack([s[k].float() for s in states]).mean(0)
    for k in keys:
        if k.endswith("running_var"):
            mk = k.replace("running_var", "running_mean")
            mi = torch.stack([s[mk].float() for s in states]); vi = torch.stack([s[k].float() for s in states])
            out[k] = (vi + mi ** 2).mean(0) - mi.mean(0) ** 2
    return out


@torch.no_grad()
def feats(bb, loader, device):
    """回傳 (vec512, z128, energy, labels, pred, vec_raw)。"""
    V, Z, E, Y, P, R = [], [], [], [], [], []
    for b in loader:
        x, y, _ = util.unpack_batch(b)
        x = x.to(device)
        h = bb.backbone.maxpool(bb.backbone.relu(bb.backbone.bn1(bb.backbone.conv1(x))))
        h = bb.backbone.layer1(h); h = bb.backbone.layer2(h); h = bb.backbone.layer3(h)
        logits, vec = bb.forward_from_layer3(h)
        V.append(torch.nn.functional.normalize(vec, dim=1).cpu().numpy())
        R.append(vec.cpu().numpy())                      # 未正規化，給隨機初始化 proj_head 用
        Z.append(bb.project(vec).cpu().numpy())          # project 內含 L2 normalize
        E.append((-torch.logsumexp(logits, 1)).cpu().numpy())   # 高 = OOD
        P.append(logits.argmax(1).cpu().numpy())
        Y.append(np.asarray(y).flatten())
    return (np.concatenate(V), np.concatenate(Z), np.concatenate(E),
            np.concatenate(Y), np.concatenate(P), np.concatenate(R))


def centers_from(feat, lab, n_cls):
    """由源域樣本算每類中心（L2 normalize）。"""
    C = []
    for c in range(n_cls):
        m = lab == c
        C.append(feat[m].mean(0) if m.sum() else np.zeros(feat.shape[1], dtype=feat.dtype))
    C = np.stack(C)
    return C / np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-8, None)


def min_angle(feat, C):
    return np.arccos(np.clip(feat @ C.T, -1 + 1e-7, 1 - 1e-7)).min(1) * DEG


def stats(s2, s3):
    """② vs ③ 的 pooled d'、AUROC、重疊面積（直方圖交集）。"""
    d = (s3.mean() - s2.mean()) / np.sqrt((s2.var(ddof=1) + s3.var(ddof=1)) / 2)
    au = roc_auc_score(np.r_[np.zeros(len(s2)), np.ones(len(s3))], np.r_[s2, s3])
    lo, hi = min(s2.min(), s3.min()), max(s2.max(), s3.max())
    bins = np.linspace(lo, hi, 60)
    h2, _ = np.histogram(s2, bins=bins, density=True); h3, _ = np.histogram(s3, bins=bins, density=True)
    ov = float(np.minimum(h2, h3).sum() * (bins[1] - bins[0]))
    return d, au, ov


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leave_out", required=True, choices=PACS)
    p.add_argument("--checkpoint_dir", required=True); p.add_argument("--description", required=True)
    p.add_argument("--datasetRoot", default="../datasets/"); p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--unknown_idx", type=int, default=6); p.add_argument("--num_nodes", type=int, default=9)
    p.add_argument("--batch_size", type=int, default=128); p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ckpt_epoch", default="final"); p.add_argument("--device", default="cuda")
    p.add_argument("--bn", choices=["avgB", "raw"], default="avgB",
                   help="BN running 統計口徑。avgB(預設,現行行為)=跨節點平均B法 mean(var+mean^2)-mean(mean)^2；"
                        "raw=各節點原樣不動。⚠️ 兩檔的數字不可相減（2026-09-06：跨 BN 口徑與跨管線是兩層不同的不可比）。")
    p.add_argument("--dump_npz", default=None,
                   help="另存逐樣本特徵給跨 session 分析：n{i}_{src,tgt}_{h,z,y} + n{i}_{C512,C128,Cema}。"
                        "h=512維penultimate(未正規化)、z=project(vec)(已含L2)、y=ImageFolder標籤(person=6)。"
                        "BN 已先做平均B。")
    a = p.parse_args()
    avail = [d for d in PACS if d != a.leave_out]
    node_src = [avail[i // (a.num_nodes // 3)] for i in range(a.num_nodes)]
    ld = {d: TD.load_pacs_test_data(a.datasetRoot, d, a.batch_size, a.num_workers)[0] for d in PACS}
    paths = [os.path.join(a.checkpoint_dir, f"{a.description}_node_{i}_{a.ckpt_epoch}.pth") for i in range(a.num_nodes)]
    if a.bn == "avgB":
        states = [torch.load(q, map_location="cpu", weights_only=False)["backbone_state"] for q in paths]
        AVG = bn_avg_B(states); del states
    else:
        AVG = {}          # raw：不覆蓋任何 BN buffer，各節點保留自己的 running 統計
    print(f"  [BN 口徑] {a.bn}" + ("（跨節點平均B）" if a.bn == "avgB" else "（各節點原樣）"))
    rows = []
    dump = {} if a.dump_npz else None
    for i, q in enumerate(paths):
        bb, _ = load_backbone_diffusion(q, a.num_classes, a.device)
        bd = dict(bb.named_buffers())
        for k, v in AVG.items():
            if k in bd: bd[k].copy_(v.to(bd[k].device).to(bd[k].dtype))
        bb.eval()
        sV, sZ, _, sY, _, sR = feats(bb, ld[node_src[i]], a.device)   # 源域（算中心用）
        tV, tZ, tE, tY, tP, tR = feats(bb, ld[a.leave_out], a.device)  # target 域
        mk = sY != a.unknown_idx
        C512 = centers_from(sV[mk], sY[mk], a.num_classes)
        C128 = centers_from(sZ[mk], sY[mk], a.num_classes)
        m2, m3 = tY != a.unknown_idx, tY == a.unknown_idx
        out = {"node": i}
        # ── 臂2：隨機正交投影 512→128（不訓練）── 區分「降維本身有害」vs「學到的權重有害」
        rng = np.random.default_rng(2026 + i)
        rp_au, rp_d, rp_ov, rp_w = [], [], [], []
        for _t in range(5):
            Wr, _ = np.linalg.qr(rng.standard_normal((sV.shape[1], 128)))   # 512x128 正交
            proj = lambda F: (F @ Wr) / np.clip(np.linalg.norm(F @ Wr, axis=1, keepdims=True), 1e-8, None)
            Crp = centers_from(proj(sV[mk]), sY[mk], a.num_classes)
            srp = min_angle(proj(tV), Crp)
            _d, _au, _ov = stats(srp[m2], srp[m3])
            rp_d.append(_d); rp_au.append(_au); rp_ov.append(_ov)
            rp_w.append((srp[m2].mean(), srp[m3].mean(),
                         srp[m2].std(ddof=1), srp[m3].std(ddof=1)))
        out["rand128"] = (np.nan, np.nan, float(np.mean(rp_d)), float(np.mean(rp_au)),
                          float(np.mean(rp_ov)), float(np.std(rp_au)))
        _ang = {"512": min_angle(tV, C512), "128": min_angle(tZ, C128)}
        out["w"] = {t: (a_[m2].mean(), a_[m3].mean(), a_[m2].std(ddof=1), a_[m3].std(ddof=1))
                    for t, a_ in _ang.items()}
        out["w"]["rand128"] = tuple(np.mean(np.array(rp_w), axis=0))
        # ── 臂2b：同架構但隨機初始化的 proj_head（Linear-ReLU-Linear，非正交）──
        _in = sR.shape[1]
        rm_au, rm_d, rm_ov, rm_w = [], [], [], []
        for _t in range(5):
            torch.manual_seed(2026 + 100 * i + _t)
            _mlp = torch.nn.Sequential(torch.nn.Linear(_in, _in), torch.nn.ReLU(),
                                       torch.nn.Linear(_in, 128)).to(a.device).eval()
            with torch.no_grad():
                _pz = lambda A: torch.nn.functional.normalize(
                    _mlp(torch.from_numpy(A).to(a.device)), dim=1).cpu().numpy()
                sM, tM = _pz(sR), _pz(tR)
            Cm = centers_from(sM[mk], sY[mk], a.num_classes)
            smm = min_angle(tM, Cm)
            _d, _au, _ov = stats(smm[m2], smm[m3])
            rm_d.append(_d); rm_au.append(_au); rm_ov.append(_ov)
            rm_w.append((smm[m2].mean(), smm[m3].mean(),
                         smm[m2].std(ddof=1), smm[m3].std(ddof=1)))
        out["randmlp128"] = (np.nan, np.nan, float(np.mean(rm_d)), float(np.mean(rm_au)),
                             float(np.mean(rm_ov)), float(np.std(rm_au)))
        out["rm_au_seeds"] = list(map(float, rm_au))   # 逐 seed，供跨 run 配對相減
        out["w"]["randmlp128"] = tuple(np.mean(np.array(rm_w), axis=0))
        out["wraw_mlp"] = rm_w      # 逐 seed 保留，供散布放大的 seed 間 std（解析度自檢）
        for tag, s in [("512", _ang["512"]), ("128", _ang["128"]), ("energy", tE)]:
            d, au, ov = stats(s[m2], s[m3])
            oscr = compute_oscr(s, tP, tY, a.unknown_idx)
            out[tag] = (s[m2].mean(), s[m3].mean(), d, au, ov, oscr)
        # 對照：checkpoint 內建的 EMA 原型（osdg_eval 用的那個）
        if hasattr(bb, "prototypes"):
            from dood.prototype import class_centers
            Cb = class_centers(bb.prototypes, bb.proto_count).cpu().numpy()
            out["buffer"] = stats(*(lambda s: (s[m2], s[m3]))(min_angle(tZ, Cb)))
            _bs = min_angle(tZ, Cb)
            out["buffer"] = (np.nan, np.nan) + out["buffer"] + (compute_oscr(_bs, tP, tY, a.unknown_idx),)
        # ── 512 維殘差四堆（扣掉六個類別中心張成的子空間後的範數）──
        Q, _ = np.linalg.qr(C512.T)                       # [512, 6] 正交基
        res = lambda F: np.linalg.norm(F - (F @ Q) @ Q.T, axis=1)
        sm2, sm3 = sY != a.unknown_idx, sY == a.unknown_idx
        r2, r3 = res(tV[m2]), res(tV[m3])
        rd, rau, rov = stats(r2, r3)          # 殘差範數當讀出：②③ 的 d'/AUROC/重疊
        # 128 維同法：扣掉 128 維六中心張成的子空間 ⇒ 與 512 殘差的唯一變因仍是投影層
        Q8, _ = np.linalg.qr(C128.T)
        res8 = lambda F: np.linalg.norm(F - (F @ Q8) @ Q8.T, axis=1)
        q2, q3 = res8(tZ[m2]), res8(tZ[m3])
        qd, qau, qov = stats(q2, q3)
        out["resid128"] = (qd, qau, qov, compute_oscr(res8(tZ), tP, tY, a.unknown_idx))
        # 「到面」的 ②③ 同空間寬度比，與「到最近中心」的對照
        out["rw"] = {"512": (r2.std(ddof=1), r3.std(ddof=1)),
                     "128": (q2.std(ddof=1), q3.std(ddof=1))}
        # ── person 的「最近中心」集中度：③ 是不是綁在單一已知類別上 ──
        #    同一份 checkpoint、同一批樣本，只換讀出空間 ⇒ 單一變因＝投影層
        def _share(F, C, m):
            am = np.argmin(np.arccos(np.clip(F @ C.T, -1 + 1e-7, 1 - 1e-7)), axis=1)[m]
            cnt = np.bincount(am, minlength=a.num_classes)
            return cnt / max(cnt.sum(), 1)
        out["conc"] = {"512": (_share(tV, C512, m3), _share(tV, C512, m2)),
                       "128": (_share(tZ, C128, m3), _share(tZ, C128, m2))}
        # ── 已知類別的「最近中心分類」與「被吸向該折最大吸引子」的比例 ──
        #    吸引子＝該折自己 target 已知樣本最常被判給的類別（不寫死類別）
        def _collapse(F, C, m, y):
            am = np.argmin(np.arccos(np.clip(F @ C.T, -1 + 1e-7, 1 - 1e-7)), axis=1)[m]
            yt = y[m]
            acc = float((am == yt).mean())
            # ⚠️ 吸引子＝「把別類樣本吸過來最多」的類別，必須**排除對角線**
            #    （用 bincount(am).argmax() 會把「自己被判對」也算進去 ⇒ 量到的是
            #      「最常被預測的類」而不是「最強吸引子」，兩者可以是不同的類）
            att_v = np.array([float((am[yt != j] == j).mean()) if (yt != j).sum() else 0.0
                              for j in range(a.num_classes)])
            att = int(att_v.argmax())
            return acc, att, float(att_v[att]), att_v
        # ① 來源域已知類別到最近中心的角度（與 ②③ 同一把尺）⇒ 壓緊有沒有遷移到 target
        _sm = sY != a.unknown_idx
        out["src_ang"] = (float(min_angle(sV[_sm], C512).mean()),
                          float(min_angle(sZ[_sm], C128).mean()))
        out["col"] = {"tgt512": _collapse(tV, C512, m2, tY),
                      "tgt128": _collapse(tZ, C128, m2, tY),
                      "src512": _collapse(sV, C512, sY != a.unknown_idx, sY),
                      "src128": _collapse(sZ, C128, sY != a.unknown_idx, sY)}
        out["resid"] = (res(sV[sm2]).mean(), res(sV[sm3]).mean(), r2.mean(), r3.mean(),
                        rd, rau, rov, compute_oscr(res(tV), tP, tY, a.unknown_idx))
        if dump is not None:
            dump[f"n{i}_src_h"], dump[f"n{i}_src_z"], dump[f"n{i}_src_y"] = sV, sZ, sY
            dump[f"n{i}_tgt_h"], dump[f"n{i}_tgt_z"], dump[f"n{i}_tgt_y"] = tV, tZ, tY
            dump[f"n{i}_C512"], dump[f"n{i}_C128"] = C512, C128
            if hasattr(bb, "prototypes"):
                from dood.prototype import class_centers
                dump[f"n{i}_Cema"] = class_centers(bb.prototypes, bb.proto_count).cpu().numpy()
        rows.append(out)
        print(f"  node_{i} ({node_src[i]:12s}) " + "  ".join(
            f"{t}:d'={rows[-1][t][2]:.3f} AUROC={rows[-1][t][3]:.4f} 重疊={rows[-1][t][4]:.3f}"
            for t in ["512", "128", "energy"]))
    print(f"\n===== {a.leave_out} node-mean =====")
    for t in ["512", "rand128", "randmlp128", "128", "energy"] + (["buffer"] if "buffer" in rows[0] else []):
        v = np.array([r[t] for r in rows], dtype=float)
        _tail = (f"  OSCR={v[:,5].mean():.4f}" if t != "rand128"
                 else f"  (5 個隨機矩陣，節點內 AUROC std={v[:,5].mean():.4f})")
        _tail = _tail if t != "randmlp128" else f"  (5 個隨機初始化 MLP，節點內 AUROC std={v[:,5].mean():.4f})"
        print(f"  {t:7s} ②={np.nanmean(v[:,0]):7.3f} ③={np.nanmean(v[:,1]):7.3f} "
              f"pooled d'={v[:,2].mean():6.3f}  AUROC={v[:,3].mean():.4f}  "
              f"重疊={v[:,4].mean():.3f}" + _tail)
    W = {t: np.array([r["w"][t] for r in rows], dtype=float).mean(0)
         for t in ["512", "rand128", "randmlp128", "128"]}
    print("\n  === 分解（節點平均、單位=度）===")
    print("  注意：『同空間寬度比』是一個空間內 3/2 的寬度（0821 的量）；"
          "『跨空間放大』是 128/512（投影層做了什麼）——兩者不同量，勿合併")
    for t in ["512", "rand128", "randmlp128", "128"]:
        mu2, mu3, sd2, sd3 = W[t]
        print(f"  {t:8s} signal(mu3-mu2)={mu3 - mu2:6.3f}  "
              f"pooled std={np.sqrt((sd2 ** 2 + sd3 ** 2) / 2):6.3f}  "
              f"2std={sd2:6.3f} 3std={sd3:6.3f}  same-space width 3/2={sd3 / sd2:5.2f}")
    # 逐畫風的訊號／散布放大——跨 run 比較（如 comp-only vs 全套）是非配對的，
    # 需要 run 內的畫風間變異當參照，才知道兩臂的差是否可解析。
    _bs = {}
    for _i, _d in enumerate(node_src):
        _bs.setdefault(_d, []).append(_i)
    for _t in ["128"]:
        _g = []
        for _d, _ix in _bs.items():
            w = np.array([rows[k]["w"][_t] for k in _ix]).mean(0)
            v = np.array([rows[k]["w"]["512"] for k in _ix]).mean(0)
            _g.append(((w[1]-w[0])/(v[1]-v[0]),
                       np.sqrt(w[2]**2+w[3]**2)/np.sqrt(v[2]**2+v[3]**2)))
        _g = np.array(_g)
        print(f"  逐畫風 {_t} 放大 ({'/'.join(_bs)})：訊號 "
              + " ".join(f"{x:.2f}" for x in _g[:,0])
              + f"  [std={_g[:,0].std(ddof=1):.3f}]   散布 "
              + " ".join(f"{x:.2f}" for x in _g[:,1])
              + f"  [std={_g[:,1].std(ddof=1):.3f}]")
    b2, b3, c2, c3 = W["512"]
    _g = np.array([[np.sqrt(w[2] ** 2 + w[3] ** 2) /
                    np.sqrt(r["w"]["512"][2] ** 2 + r["w"]["512"][3] ** 2)
                    for w in r["wraw_mlp"]] for r in rows])          # [node, seed]
    print(f"  randmlp128 spread gain per-seed: mean={_g.mean():5.3f}  "
          f"std(within-node, across 5 seeds)={_g.std(axis=1).mean():5.3f}  "
          f"std(across nodes)={_g.mean(axis=1).std():5.3f}  "
          f"min={_g.min():5.3f} max={_g.max():5.3f}")
    for t in ["rand128", "randmlp128", "128"]:
        mu2, mu3, sd2, sd3 = W[t]
        print(f"  {t:8s} vs 512 cross-space gain: signal={(mu3 - mu2) / (b3 - b2):5.2f}x  "
              f"spread={np.sqrt(sd2 ** 2 + sd3 ** 2) / np.sqrt(c2 ** 2 + c3 ** 2):5.2f}x")
    _M = np.array([r["rm_au_seeds"] for r in rows])   # [node, seed]
    print("  randmlp128 per-seed AUROC [node x seed] (跨 run 同 seed ⇒ 可配對相減):")
    for _i, _row in enumerate(_M):
        print("    node%d " % _i + " ".join(f"{v:.4f}" for v in _row))
    # ⚠️ 誤差的獨立單位是「畫風」不是「節點」：BN 平均 B 口徑下同畫風三節點的特徵
    #    逐樣本餘弦 >0.9998（0904 §0.2A）⇒ 9 節點只有 3 個獨立觀測點。
    _by_style = {}
    for _i, _d in enumerate(node_src):
        _by_style.setdefault(_d, []).append(_i)
    _S = np.array([_M[_ix].mean(0) for _ix in _by_style.values()])   # [style, seed]
    _sm = _S.mean(1)
    print(f"  randmlp128 逐畫風平均 ({'/'.join(_by_style)}) = "
          + " ".join(f"{v:.4f}" for v in _sm))
    print(f"  ⇒ 以 3 獨立畫風算：mean={_sm.mean():.4f}  "
          f"std(across styles)={_sm.std(ddof=1):.4f}  SE={_sm.std(ddof=1)/np.sqrt(len(_sm)):.4f}"
          f"   (⚠️ std(across nodes)={_M.mean(1).std(ddof=1):.4f} 會低估)")
    if dump is not None:
        np.savez_compressed(a.dump_npz, **dump)
        print(f"  [dump] {a.dump_npz}  keys={len(dump)}  "
              f"({os.path.getsize(a.dump_npz)/1e6:.1f} MB)")
    r = np.array([x["resid"] for x in rows], dtype=float).mean(0)
    print(f"  [512維殘差‖v⊥‖ 四堆] 源域已知={r[0]:.4f}  源域person={r[1]:.4f}  "
          f"target已知={r[2]:.4f}  target person={r[3]:.4f}")
    print(f"  [殘差當讀出·512] pooled d'={r[4]:.3f}  AUROC={r[5]:.4f}  重疊={r[6]:.3f}  OSCR={r[7]:.4f}")
    q = np.array([x["resid128"] for x in rows], dtype=float).mean(0)
    print(f"  [殘差當讀出·128] pooled d'={q[0]:.3f}  AUROC={q[1]:.4f}  重疊={q[2]:.3f}  OSCR={q[3]:.4f}")
    print("  === person 最近中心的集中度（③ person / ② 已知）===")
    for t in ["512", "128"]:
        p3 = np.array([r["conc"][t][0] for r in rows]).mean(0)
        p2 = np.array([r["conc"][t][1] for r in rows]).mean(0)
        print(f"  {t:4s} person max={p3.max():.3f} (cls {p3.argmax()})  "
              f"known max={p2.max():.3f} (cls {p2.argmax()})  ratio={p3.max()/max(p2.max(),1e-9):.2f}")
        print(f"       person per-class " + " ".join(f"{v:.3f}" for v in p3))
    _sa = np.array([r["src_ang"] for r in rows]).mean(0)
    print(f"  === ① 來源域已知到最近中心：512 {_sa[0]:.3f}°  128 {_sa[1]:.3f}° ===")
    print("  === 已知類別：最近中心分類正確率 / 被吸向最大吸引子的比例（隨機基準 %.3f）===" % (1/a.num_classes))
    for t in ["src512", "src128", "tgt512", "tgt128"]:
        v = np.array([r["col"][t][:3] for r in rows], dtype=float)
        att = int(np.bincount([int(r["col"][t][1]) for r in rows]).argmax())
        pn = np.array([r["col"][t][2] for r in rows])
        st = np.array([pn[[i for i, d in enumerate(node_src) if d == dd]].mean()
                       for dd in dict.fromkeys(node_src)])
        print(f"  {t:7s} acc={v[:,0].mean():.3f}  吸引子=cls{att}  被吸比例={pn.mean():.3f}"
              f"  逐畫風 {' '.join(f'{x:.3f}' for x in st)}  std={st.std(ddof=1):.3f}")
        full = np.array([r["col"][t][3] for r in rows]).mean(0)
        print(f"          逐類別吸引量 " + " ".join(f"c{j}:{x:.3f}" for j, x in enumerate(full)))
    RW = {t: np.array([x["rw"][t] for x in rows], dtype=float).mean(0) for t in ["512", "128"]}
    print("  === point-vs-plane: same-space width ratio 2/3 ===")
    for t in ["512", "128"]:
        a2, a3, b2, b3 = W[t][2], W[t][3], RW[t][0], RW[t][1]
        print(f"  {t:4s} nearest-center angle 2/3={a2 / a3:5.2f}   "
              f"residual ||v_perp|| 2/3={b2 / b3:5.2f}")


if __name__ == "__main__":
    main()
