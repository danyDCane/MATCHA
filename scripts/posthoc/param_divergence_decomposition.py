"""9 節點終態 checkpoint 的參數分歧拆解 + 原型漂移的逐域對拆解。零前向。"""
import os,sys,torch,numpy as np,collections
sys.path.insert(0,"scripts"); sys.path.insert(0,".")
DESC="v1_stage2_leave_cartoon_p1a_async_const_tau1e-5_style_nodiff_aggbn_osdg_excl_person_seed2026_topo1234_fix"
CK="exp_result_"+DESC
OWN=["art"]*3+["photo"]*3+["sketch"]*3
SD=[torch.load(os.path.join(CK,f"{DESC}_node_{i}_final.pth"),map_location="cpu",weights_only=False) for i in range(9)]
SD=[s["backbone_state"] for s in SD]
keys=list(SD[0].keys())
print("checkpoint 頂層鍵樣本：",keys[:6],"...共",len(keys))

def group(k):
    if "prototype" in k or "proto_count" in k: return "原型 buffer"
    if "num_batches_tracked" in k: return "(略)"
    if "running_mean" in k or "running_var" in k: return "★BN running 統計量"
    if ("bn" in k or "downsample.1" in k) and (k.endswith(".weight") or k.endswith(".bias")): return "BN 仿射 γ/β"
    if "proj_head" in k: return "投影層"
    if k.startswith("fc") or "classifier" in k or "linear" in k: return "分類頭"
    return "conv 權重"

G=collections.defaultdict(list)
for k in keys:
    g=group(k)
    if g=="(略)": continue
    V=[SD[i][k].float() for i in range(9)]
    if V[0].numel()==0: continue
    mu=torch.stack(V).mean(0)
    nm=mu.norm().item()
    if nm<1e-12: continue
    G[g].append(float(np.mean([ (v-mu).norm().item() for v in V])/nm))
print("\n"+"="*72); print("【分歧度】mean‖v_i − 平均‖ / ‖平均‖（與 BN-DIV 同公式，可直接對照）"); print("="*72)
for g in ["conv 權重","BN 仿射 γ/β","分類頭","投影層","★BN running 統計量","原型 buffer"]:
    if g in G: print(f"  {g:<22}{np.mean(G[g]):>12.6f}   （{len(G[g])} 個張量）")
print("\n  對照：0730 cartoon BN 各自時 BN-DIV = 0.345；0805 KSD run conv/fc≈5e-09、norm buffer≈1.1e-01")

# 原型漂移逐域對
P=[]
for i in range(9):
    pr=SD[i]["prototypes"].float() if "prototypes" in SD[i] else None
    pc=SD[i]["proto_count"].float() if "proto_count" in SD[i] else None
    C=(pr/pc.clamp(min=1).unsqueeze(1)) if pc is not None and pr.dim()==2 else pr
    C=C/C.norm(dim=1,keepdim=True); P.append(C.numpy())
PD=np.zeros((9,9))
for i in range(9):
    for j in range(9):
        PD[i,j]=np.degrees(np.arccos(np.clip((P[i]*P[j]).sum(1),-1,1))).mean()
print("\n"+"="*72); print("【原型漂移】同類別原型在不同節點之間的夾角（度）"); print("="*72)
print("      "+"".join(f"{'nd'+str(j):>7}" for j in range(9)))
for i in range(9):
    print(f"nd{i} {OWN[i][:5]:<4}"+"".join(f"{PD[i,j]:>7.2f}" for j in range(9)))
print()
pairs=collections.defaultdict(list)
for i in range(9):
    for j in range(i+1,9):
        pairs[("同風格 "+OWN[i]) if OWN[i]==OWN[j] else " ↔ ".join(sorted([OWN[i],OWN[j]]))].append(PD[i,j])
print("按節點對分類：")
for k in sorted(pairs,key=lambda x:(("同風格" not in x),x)):
    print(f"  {k:<22}{np.mean(pairs[k]):>8.2f}°   （{len(pairs[k])} 對，範圍 {min(pairs[k]):.2f}–{max(pairs[k]):.2f}）")
same=[v for k,vs in pairs.items() if "同風格" in k for v in vs]
diff=[v for k,vs in pairs.items() if "同風格" not in k for v in vs]
print(f"\n  同風格全部 {np.mean(same):.2f}°  ｜  跨風格全部 {np.mean(diff):.2f}°  ｜  比值 {np.mean(diff)/np.mean(same):.2f}x")
