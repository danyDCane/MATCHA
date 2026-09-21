"""dany 2026-08-29 的質疑：26% 覆蓋率是在【已訓練模型】上做 AdaIN 介入量的，
若模型對 channel 統計量已經免疫，介入本來就沒效 ⇒ 26% 低不代表「該軸只佔 26%」。
本腳本用落盤的 layer1/2/3 通道統計量做四項交叉檢驗（post-hoc、零重訓）。
基底：0826_features_full.npz（BN 平均 B）。
"""
import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression
from scipy.stats import rankdata
F = np.load("logs/prototype_probe/0826_features_full.npz"); N, UNK = 9, 6
nrm = lambda X: X/np.linalg.norm(X, axis=-1, keepdims=True)
def auroc(pos, neg):
    a=np.concatenate([pos,neg]); r=rankdata(a,method="average")
    return float((r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))

D={}
for i in range(N):
    C=nrm(F[f"n{i}_C"].astype(np.float64)); U=np.linalg.svd(C.T,full_matrices=False)[0]
    d={}
    for k in ["src","tgt"]:
        Z=nrm(F[f"n{i}_{k}_z"].astype(np.float64))
        S=np.concatenate([F[f"n{i}_{k}_s{j}"].astype(np.float64) for j in [1,2,3]],1)
        d[k]=dict(zp=np.sqrt(np.maximum(1-((Z@U)**2).sum(1),0)), S=S, y=F[f"n{i}_{k}_y"])
    D[i]=d
print("="*100); print("通道統計量維度：", D[0]["src"]["S"].shape[1], "（layer1/2/3 各 mean+std）"); print("="*100)

print("\n【檢驗A】通道統計量能不能分辨畫風？（能 ⇒ 它確實攜帶大量畫風資訊）")
a=[]
for i in range(N):
    s,t=D[i]["src"],D[i]["tgt"]; m1,m2=s["y"]!=UNK, t["y"]!=UNK
    X=np.vstack([s["S"][m1],t["S"][m2]]); yy=np.r_[np.zeros(m1.sum()),np.ones(m2.sum())]
    mu,sd=X.mean(0),X.std(0)+1e-9; X=(X-mu)/sd
    idx=np.arange(len(X)); rng=np.random.default_rng(i); rng.shuffle(idx)
    tr,te=idx[:int(.7*len(idx))],idx[int(.7*len(idx)):]
    clf=LogisticRegression(C=0.01,max_iter=2000).fit(X[tr],yy[tr])
    p=clf.decision_function(X[te]); a.append(auroc(p[yy[te]==1],p[yy[te]==0]))
print(f"  來源域 vs cartoon 的可分性（held-out 30%）：AUROC = {np.mean(a):.4f}")
print(f"  ⇒ {'通道統計量【確實】強烈編碼畫風' if np.mean(a)>0.9 else '通道統計量對畫風的編碼有限'}")

print("\n【檢驗B/C】★ 通道統計量能解釋 ‖z⊥‖ 的畫風增長多少？")
print("  做法：只用【來源域已知類別】訓 ridge（‖z⊥‖ ~ 通道統計量），再看它對 cartoon 的預測")
r2,gap_true,gap_pred=[],[],[]
for i in range(N):
    s,t=D[i]["src"],D[i]["tgt"]; m1,m2=s["y"]!=UNK,t["y"]!=UNK
    X1,y1=s["S"][m1],s["zp"][m1]; X2,y2=t["S"][m2],t["zp"][m2]
    mu,sd=X1.mean(0),X1.std(0)+1e-9
    idx=np.arange(len(X1)); rng=np.random.default_rng(i); rng.shuffle(idx)
    tr,te=idx[:int(.7*len(idx))],idx[int(.7*len(idx)):]
    rg=RidgeCV(alphas=np.logspace(-1,4,12)).fit((X1[tr]-mu)/sd,y1[tr])
    p_te=rg.predict((X1[te]-mu)/sd); p_tgt=rg.predict((X2-mu)/sd)
    r2.append(1-((y1[te]-p_te)**2).sum()/((y1[te]-y1[te].mean())**2).sum())
    gap_true.append(y2.mean()-y1[te].mean()); gap_pred.append(p_tgt.mean()-p_te.mean())
print(f"  來源域內的解釋力 R²(held-out) = {np.mean(r2):.4f}")
print(f"  ‖z⊥‖ 的畫風增長：實測 {np.mean(gap_true):+.4f}   通道統計量預測得出 {np.mean(gap_pred):+.4f}"
      f"   ⇒ 解釋 {np.mean(gap_pred)/np.mean(gap_true)*100:.1f}%")

print("\n【檢驗D】★★ 直接測 dany 的 confound：模型對通道統計量到底敏不敏感？")
print("  做法：把 cartoon 樣本按「通道統計量離來源域中心多遠」分五層，看 ‖z⊥‖ 是否隨之上升")
print("  零點對照：改用【隨機方向上的投影】分層（無資訊），看是否也出現同樣的梯度")
rows,rows0=[],[]
rng=np.random.default_rng(2026)
for i in range(N):
    s,t=D[i]["src"],D[i]["tgt"]; m1,m2=s["y"]!=UNK,t["y"]!=UNK
    mu,sd=s["S"][m1].mean(0),s["S"][m1].std(0)+1e-9
    dist=np.linalg.norm((t["S"][m2]-mu)/sd,axis=1)
    w=rng.standard_normal(t["S"].shape[1]); rnd=np.abs(((t["S"][m2]-mu)/sd)@w/np.linalg.norm(w))
    zp=t["zp"][m2]
    for arr,acc in [(dist,rows),(rnd,rows0)]:
        q=np.quantile(arr,np.linspace(0,1,6)); q[0]-=1e-9
        dec=np.clip(np.searchsorted(q,arr,side="left")-1,0,4)
        acc.append([zp[dec==d].mean() for d in range(5)])
R,R0=np.mean(rows,0),np.mean(rows0,0)
print(f"    {'層(離來源域統計量由近到遠)':<28}{'1':>8}{'2':>8}{'3':>8}{'4':>8}{'5':>8}{'  跨度':>8}")
print(f"    {'★ 按通道統計量距離分層':<28}"+"".join(f"{v:8.4f}" for v in R)+f"{R[-1]-R[0]:8.4f}")
print(f"    {'零點：按隨機方向分層':<28}"+"".join(f"{v:8.4f}" for v in R0)+f"{R0[-1]-R0[0]:8.4f}")
print(f"    ⇒ 真實梯度是零點的 {(R[-1]-R[0])/abs(R0[-1]-R0[0]+1e-12):.1f} 倍"
      f"；而 ①→② 的總畫風增長是 0.1173 ⇒ 層間跨度佔其 {(R[-1]-R[0])/0.1173*100:.0f}%")
print("="*100)
