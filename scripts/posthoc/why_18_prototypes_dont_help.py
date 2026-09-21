"""18 個原型取 min 為什麼幾乎不動 AUROC？拆成四環，每環給數字。
基底：logs/prototype_probe/0826_features_full.npz（BN 平均 B，由 probe_information_content.py 抽）
對照：0819b §6 的 P0 .8145 / P1 .8151 / P2 .8148
"""
import numpy as np
F = np.load("logs/prototype_probe/0826_features_full.npz")
DEG = 57.29577951308232; N, NC, UNK = 9, 6, 6
AVAIL = ["art", "photo", "sketch"]; OWN = [AVAIL[i // 3] for i in range(N)]
nrm = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)

def auroc(pos, neg):           # pos=person（分數該高）
    a = np.concatenate([pos, neg]); r = a.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

CEN = nrm(np.stack([F[f"n{i}_C"] for i in range(N)]).astype(np.float64))      # [9,6,128]
P1  = nrm(CEN.mean(0))                                                        # [6,128] 9節點平均
P3  = nrm(np.stack([CEN[[i for i in range(N) if OWN[i] == d]].mean(0) for d in AVAIL]))  # [3,6,128]
P18 = P3.transpose(1, 0, 2).reshape(NC * 3, -1)                               # [18,128]
LAB18 = np.repeat(np.arange(NC), 3)

print("="*86); print("§0 自檢：重現 0819b §6 的三種讀出（node-mean 部署 AUROC）"); print("="*86)
res = {k: [] for k in ["P0", "P1", "P2"]}; d2 = {k: [] for k in ["P0","P1","P2"]}; d3 = {k: [] for k in ["P0","P1","P2"]}
for i in range(N):
    Z = nrm(F[f"n{i}_tgt_z"].astype(np.float64)); Y = F[f"n{i}_tgt_y"]
    m2, m3 = Y != UNK, Y == UNK
    for k, C in [("P0", CEN[i]), ("P1", P1), ("P2", P18)]:
        s = np.arccos(np.clip(Z @ C.T, -1+1e-12, 1-1e-12)).min(1) * DEG
        res[k].append(auroc(s[m3], s[m2])); d2[k].append(s[m2].mean()); d3[k].append(s[m3].mean())
for k, ref in [("P0", .8145), ("P1", .8151), ("P2", .8148)]:
    print(f"  {k}  部署 AUROC {np.mean(res[k]):.4f}   （0819b 報 {ref:.4f}）"
          f"   ②{np.mean(d2[k]):5.2f}°  ③{np.mean(d3[k]):5.2f}°  差 {np.mean(d3[k])-np.mean(d2[k]):5.2f}°")

print(); print("="*86); print("環節1：三個畫風的原型有多近？（dany 的猜測）"); print("="*86)
G = np.arccos(np.clip(P18 @ P18.T, -1, 1)) * DEG
same = [G[a, b] for a in range(18) for b in range(a+1, 18) if LAB18[a] == LAB18[b]]
diff = [G[a, b] for a in range(18) for b in range(a+1, 18) if LAB18[a] != LAB18[b]]
print(f"  同一類、不同畫風的原型    平均 {np.mean(same):6.2f}°   範圍 {np.min(same):5.2f}–{np.max(same):5.2f}°  (n={len(same)})")
print(f"  不同類別的原型            平均 {np.mean(diff):6.2f}°   最小 {np.min(diff):5.2f}°          (n={len(diff)})")
print(f"  ⇒ 比值 {np.mean(diff)/np.mean(same):.1f}×；同類三原型的分散只有跨類間距的 {np.mean(same)/np.mean(diff)*100:.1f}%")

print(); print("="*86); print("環節2+3：加了 12 個中心，②與③ 各降多少？（AUROC 只看排序）"); print("="*86)
D2, D3, FLIP = [], [], []
for i in range(N):
    Z = nrm(F[f"n{i}_tgt_z"].astype(np.float64)); Y = F[f"n{i}_tgt_y"]
    s1 = np.arccos(np.clip(Z @ P1.T, -1+1e-12, 1-1e-12)).min(1) * DEG
    s2 = np.arccos(np.clip(Z @ P18.T, -1+1e-12, 1-1e-12)).min(1) * DEG
    m2, m3 = Y != UNK, Y == UNK
    D2.append((s1-s2)[m2].mean()); D3.append((s1-s2)[m3].mean())
    r1, r2 = s1.argsort().argsort(), s2.argsort().argsort()
    FLIP.append(np.mean(np.abs(r1-r2) > 0.01*len(s1)))
print(f"  ② cartoon 已知類別   分數平均降 {np.mean(D2):.3f}°")
print(f"  ③ person             分數平均降 {np.mean(D3):.3f}°")
print(f"  ⇒ 兩堆的降幅差只有 {abs(np.mean(D3)-np.mean(D2)):.3f}°  ＝ 幾乎是共同位移（排名不動）")
print(f"  ⇒ 排名移動超過全體 1% 名次的樣本比例：{np.mean(FLIP)*100:.1f}%")

print(); print("="*86); print("環節4：18 個原型張成幾維？能不能多描述一點 person？"); print("="*86)
sv = np.linalg.svd(P18, compute_uv=False); e = sv**2 / (sv**2).sum()
c = np.cumsum(e)
print(f"  18 個原型的奇異值能量：前6維 {c[5]*100:.1f}%   前9維 {c[8]*100:.1f}%   到 90% 需 {np.argmax(c>=.90)+1} 維  到 99% 需 {np.argmax(c>=.99)+1} 維")
U6  = np.linalg.svd(P1.T,  full_matrices=False)[0]
U18 = np.linalg.svd(P18.T, full_matrices=False)[0][:, :np.argmax(c>=.99)+1]
print(f"  P18 相對 P1(6維) 的新增有效維度：{U18.shape[1]-6} 維（99% 能量門檻）")
E2, E3, E2b, E3b = [], [], [], []
for i in range(N):
    Z = nrm(F[f"n{i}_tgt_z"].astype(np.float64)); Y = F[f"n{i}_tgt_y"]
    m2, m3 = Y != UNK, Y == UNK
    for U, a, b in [(U6, E2, E3), (U18, E2b, E3b)]:
        p = ((Z @ U)**2).sum(1)                     # ‖z∥‖²，因 ‖z‖=1
        a.append(p[m2].mean()); b.append(p[m3].mean())
print(f"  六個原型(6維)  解釋 ② {np.mean(E2)*100:5.1f}%   解釋 ③ {np.mean(E3)*100:5.1f}%   差 {(np.mean(E2)-np.mean(E3))*100:.1f}pp")
print(f"  18 個原型({U18.shape[1]}維) 解釋 ② {np.mean(E2b)*100:5.1f}%   解釋 ③ {np.mean(E3b)*100:5.1f}%   差 {(np.mean(E2b)-np.mean(E3b))*100:.1f}pp")
print("="*86)

print(); print("="*86); print("補：記帳確認（0825 §5.6 報 ②60.5% / ③27.4%，那組用的是各節點自己的原型）"); print("="*86)
for tag, get in [("P0 各節點自己的 6 個", lambda i: CEN[i]), ("P1 9節點平均的 6 個", lambda i: P1)]:
    a, b = [], []
    for i in range(N):
        Z = nrm(F[f"n{i}_tgt_z"].astype(np.float64)); Y = F[f"n{i}_tgt_y"]; m2, m3 = Y != UNK, Y == UNK
        U = np.linalg.svd(get(i).T, full_matrices=False)[0]
        p = ((Z @ U)**2).sum(1); a.append(p[m2].mean()); b.append(p[m3].mean())
    print(f"  {tag}：解釋 ② {np.mean(a)*100:5.1f}%   解釋 ③ {np.mean(b)*100:5.1f}%   差 {(np.mean(a)-np.mean(b))*100:.1f}pp")
a, b = [], []
U18f = np.linalg.svd(P18.T, full_matrices=False)[0]
for i in range(N):
    Z = nrm(F[f"n{i}_tgt_z"].astype(np.float64)); Y = F[f"n{i}_tgt_y"]; m2, m3 = Y != UNK, Y == UNK
    p = ((Z @ U18f)**2).sum(1); a.append(p[m2].mean()); b.append(p[m3].mean())
print(f"  P18 完整 span（{U18f.shape[1]} 維、不做能量截斷）：解釋 ② {np.mean(a)*100:5.1f}%   解釋 ③ {np.mean(b)*100:5.1f}%   差 {(np.mean(a)-np.mean(b))*100:.1f}pp")
print("="*86)

print(); print("="*86); print("環節4b：無資訊基準——「6 維原型 ＋ 殘差裡隨機 12 維」也會讓解釋率上升多少？"); print("="*86)
rng = np.random.default_rng(2026)
U6f = np.linalg.svd(P1.T, full_matrices=False)[0]                       # [128,6]
Q = np.linalg.svd(np.eye(128) - U6f @ U6f.T)[0][:, :122]                # 殘差正交基
ZS = [(nrm(F[f"n{i}_tgt_z"].astype(np.float64)), F[f"n{i}_tgt_y"]) for i in range(N)]
def explain(U):
    a, b = [], []
    for Z, Y in ZS:
        p = ((Z @ U)**2).sum(1); a.append(p[Y != UNK].mean()); b.append(p[Y == UNK].mean())
    return np.mean(a), np.mean(b)
e2_0, e3_0 = explain(U6f)
R2, R3 = [], []
for _ in range(200):
    W = np.linalg.svd(Q @ rng.standard_normal((122, 12)), full_matrices=False)[0]
    x, y = explain(np.hstack([U6f, W])); R2.append(x - e2_0); R3.append(y - e3_0)
e2_18, e3_18 = explain(U18f)
print(f"  基準線（只有 6 維原型）        ② {e2_0*100:5.1f}%   ③ {e3_0*100:5.1f}%")
print(f"  加真實的 12 個原型方向  增量   ② {(e2_18-e2_0)*100:+5.2f}pp   ③ {(e3_18-e3_0)*100:+5.2f}pp")
print(f"  加隨機的 12 個殘差方向  增量   ② {np.mean(R2)*100:+5.2f}±{np.std(R2)*100:.2f}pp   "
      f"③ {np.mean(R3)*100:+5.2f}±{np.std(R3)*100:.2f}pp   (200 次)")
for nm_, real, rnd in [("②", e2_18-e2_0, R2), ("③", e3_18-e3_0, R3)]:
    z = (real - np.mean(rnd)) / np.std(rnd)
    print(f"  ⇒ {nm_} 真實增量相對隨機基準：{z:+.2f}σ   百分位 {np.mean(np.array(rnd) < real)*100:5.1f}%")
print("="*86)
