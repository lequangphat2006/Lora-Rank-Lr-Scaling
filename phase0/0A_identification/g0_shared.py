"""g0_shared.py - Estimator 'be mat chia se' (shared-shape) va so sanh voi naive/mask tren cung 9 be mat.
Mo hinh: loss(x,r,p) = m_{r,p} + f(x - t(r,p)), f(d)=c2 d^2 + c3 d^3 + c4 d^4 CHUNG moi (r,p); m_{r,p} rieng tung duong cong;
t(r,p) = b0 + b1*log(r/8) + b2*p*log(r/8)  =>  a=-b1, b=b2. Khop bien-phan (variable projection) + Levenberg-Marquardt theo lo.
Diem collapse bi loai (trong so 0). Khoi dau tu uoc luong dinh parabol (mask)."""
import numpy as np, sys, g0_final as G, g0_robustness as R
from multiprocessing import Pool
P=G.P; L=G.L
def _res(th, y, W, xg):
    t=th[:,0,None,None]+th[:,1,None,None]*L[None,None,:]+th[:,2,None,None]*(P[None,:,None]*L[None,None,:])   # (N,3,7)
    d=xg[None,None,None,:]-t[...,None]                                                                        # (N,3,7,m)
    Phi=np.stack([d**2,d**3,d**4],-1)
    sw=np.maximum(W.sum(-1,keepdims=True),1.0)
    ym=(W*y).sum(-1,keepdims=True)/sw; Pm=(W[...,None]*Phi).sum(-2,keepdims=True)/sw[...,None]
    yd=y-ym; Pd=Phi-Pm
    A=np.einsum('nprmi,nprm,nprmj->nij',Pd,W,Pd); A=A+1e-6*(np.einsum('nii->n',A)/3+1e-12)[:,None,None]*np.eye(3); r=np.einsum('nprmi,nprm,nprm->ni',Pd,W,yd)
    c=np.linalg.solve(A,r[...,None])[...,0]
    res=W*(yd-np.einsum('nprmi,ni->nprm',Pd,c))
    return res.reshape(len(y),-1), c
def refine(y,W,xg,th0,iters=12):
    th=th0.copy(); res,_=_res(th,y,W,xg); sse=(res**2).sum(1); lam=np.full(len(y),1e-2); h=1e-4
    for _ in range(iters):
        J=np.stack([(_res(th+h*np.eye(3)[i],y,W,xg)[0]-res)/h for i in range(3)],-1)       # (N,K,3)
        JTJ=np.einsum('nki,nkj->nij',J,J); g=np.einsum('nki,nk->ni',J,res)
        D=np.einsum('nii->ni',JTJ)+1e-9
        step=np.linalg.solve(JTJ+lam[:,None,None]*np.einsum('ni,ij->nij',D,np.eye(3)),-g[...,None])[...,0]
        thn=np.clip(np.nan_to_num(th+step,nan=0.0),[-6,-3,-3],[-1,3,3])
        rn,_=_res(thn,y,W,xg); sn=np.nan_to_num((rn**2).sum(1),nan=np.inf); ok=sn<sse
        th=np.where(ok[:,None],thn,th); res=np.where(ok[:,None],rn,res); sse=np.where(ok,sn,sse)
        lam=np.where(ok,np.maximum(lam*0.3,1e-7),np.minimum(lam*5,1e6))
    return th
def est_shared(h,N,c2,n,xg,surf,rng,cliff=5.0):
    y,col=R.batch(*R.H3[h],N,c2,n,xg,surf[0],surf[1],cliff,rng)
    W=np.broadcast_to((~col).astype(float),y.shape[1:]); W=np.broadcast_to(W,y.shape)
    e=R.vertex(y,W,xg); e=np.swapaxes(e,-1,-2).reshape(e.shape[:-2]+(-1,)); co=e@G.X_PINV.T      # khoi dau
    th=refine(y,W,xg,co.copy())
    return np.stack([-th[:,1],th[:,2]],-1)
def run(args):
    step,n,c2,mode,seed,NC=args; rng=np.random.default_rng(seed+G.SEED_OFFSET); xg=G.make_grid(step)
    f=(lambda h,s:est_shared(h,NC,c2,n,xg,s,rng)) if mode=="shared" else (lambda h,s:R.est(h,NC,c2,n,xg,s,mode,rng))
    pool={s:{h:f(h,s) for h in R.H3} for s in R.SURF}; test={s:{h:f(h,s) for h in R.H3} for s in R.SURF}
    single={h:R.qfit(pool[R.ASSUMED][h]) for h in R.H3}
    mix={h:R.qfit(np.concatenate([pool[s][h] for s in R.TRAIN_MIX])) for h in R.H3}
    out={"single":{},"mixture":{},"oracle":{}}
    for s in R.SURF:
        out["single"][s]=R.acc(test[s],single); out["oracle"][s]=R.acc(test[s],{h:R.qfit(pool[s][h]) for h in R.H3})
        if s not in R.TRAIN_MIX: out["mixture"][s]=R.acc(test[s],mix)
    bias={s:np.round(np.median(test[s]["k1"],0)-np.array(R.H3["k1"]),3).tolist() for s in R.SURF}
    return step,n,c2,mode,out,bias
if __name__=="__main__":
    scale=float(sys.argv[1]) if len(sys.argv)>1 else 1.0; NC=int(800*scale); jobs=[];k=0
    for step,n in [(2.0,5),(2**.5,5),(2**.5,8)]:
        for c2 in [0.047,0.1,0.2,0.5]:
            for mode in ["mask","shared"]:
                jobs.append((step,n,c2,mode,52000+k,NC)); k+=1
    with Pool(G.NPROC) as p: res=p.map(run,jobs)
    print("Muc tieu: mixture worst >= 0.9 (hieu chinh tren 4 be mat, kiem tren 5 be mat khac). bias = (a,b) cua k1; do TREN CAC 9 be mat: [min..max]\n")
    print("step  n  c2     mode   | single mean/worst | mixture mean/worst | oracle mean/worst | bias_a [min,max]   bias_b [min,max]")
    for step,n,c2,mode,o,b in res:
        fm=lambda d:(np.mean([np.mean(list(v.values())) for v in d.values()]),min(np.mean(list(v.values())) for v in d.values()))
        sm,sw=fm(o["single"]); mm,mw=fm(o["mixture"]); om,ow=fm(o["oracle"])
        ba=[v[0] for v in b.values()]; bb=[v[1] for v in b.values()]
        print(f"{step:<5.2f}{n:<3}{c2:<7}{mode:<7}| {sm:.3f} / {sw:.3f}     | {mm:.3f} / {mw:.3f} {'PASS' if mw>=.9 else 'FAIL'} | {om:.3f} / {ow:.3f}     | [{min(ba):+.2f},{max(ba):+.2f}]  [{min(bb):+.2f},{max(bb):+.2f}]")
