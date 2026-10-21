"""g0_robustness.py - Kiem tra DO BEN cua G0 khi mo hinh be mat bi sai.
Hieu chinh QDA tren mot (hoac mot nua) ho be mat, kiem tren cac be mat KHAC.
Ho be mat: do cong nhanh trai (plateau) rho in {1,.5,.25} x dich vach collapse shift in {-.3,0,+.3} dex.
Chi 3 gia thuyet G0. Tieu chi: acc3 >= 0.9 tren MOI be mat kiem."""
import numpy as np, sys, itertools, g0_final as G
from multiprocessing import Pool
P=G.P; L=G.L; XREF=G.XREF; SIG=0.0136
H3={"k1":(1,1),"muA":(.5,.5),"khalf":(.5,1)}
SURF=list(itertools.product([1.0,0.5,0.25],[-0.3,0.0,0.3]))     # (rho, shift)
ASSUMED=(0.5,0.0)
TRAIN_MIX=[(0.5,0.0),(1.0,0.0),(0.5,0.3),(0.25,-0.3)]
def batch(a,b,N,c2,n,xg,rho,shift,cliff,rng):
    xs=XREF-(a-b*P[:,None])*L[None,:]; d=xg[None,None,:]-xs[...,None]
    base=np.where(d<0,rho*c2*d**2,c2*d**2)
    col=xg[None,:]>np.log10(1e-3)-0.5*L[:,None]+shift
    base=base+cliff*col[None]
    return base[None]+rng.normal(0,SIG/np.sqrt(n),(N,)+base.shape),col
def vertex(y,w,xg):
    xc=xg-xg.mean(); V=np.vander(xc,3); w=np.broadcast_to(w,y.shape).astype(float)
    A=np.einsum('mi,...m,mj->...ij',V,w,V)+1e-12*np.eye(3); r=np.einsum('mi,...m->...i',V,w*y)
    c=np.linalg.solve(A,r[...,None])[...,0]; ok=c[...,0]>0; s=np.where(ok,c[...,0],1)
    fb=xg[np.argmin(np.where(w>0,y,np.inf),-1)]-xg.mean()
    return np.where(ok,-c[...,1]/(2*s),fb)+xg.mean()
def est(h,N,c2,n,xg,surf,mode,rng,cliff=5.0):
    y,col=batch(*H3[h],N,c2,n,xg,surf[0],surf[1],cliff,rng)
    w=np.ones(col.shape) if mode=="naive" else (~col).astype(float)
    e=vertex(y,w,xg); e=np.swapaxes(e,-1,-2).reshape(e.shape[:-2]+(-1,)); co=e@G.X_PINV.T
    return np.stack([-co[...,1],co[...,2]],-1)
from scipy.stats import chi2 as _chi2
def rfit(X,keep=0.9,iters=8):
    """Gaussian BEN voi ngoai le: uoc luong cat tia lap (MCD-nhe) + hieu chinh nhat quan. Tra (mean, C)."""
    X=np.asarray(X,float); X=X[np.all(np.isfinite(X),1)]; p=X.shape[1]
    m=np.median(X,0); mad=1.4826*np.median(np.abs(X-m),0)+1e-9; C=np.diag(mad**2)
    q=_chi2.ppf(keep,p); fac=keep/_chi2.cdf(q,p+2)
    for _ in range(iters):
        d=np.einsum('ni,ij,nj->n',X-m,np.linalg.inv(C),X-m); idx=d<=np.quantile(d,keep)
        m=X[idx].mean(0); C=np.cov(X[idx].T)*fac+1e-10*np.eye(p)
    return m,C
def qfit(pts):
    m,C=rfit(pts); return m,np.linalg.inv(C),np.log(np.linalg.det(C))
def sc(pt,q): m,Ci,ld=q; d=pt-m; return -0.5*np.einsum('ni,ij,nj->n',d,Ci,d)-0.5*ld
def acc(test,cal):
    names=list(H3); a={}
    for k in names:
        S=np.stack([sc(test[k],cal[j]) for j in names],1); a[k]=float(np.mean(np.array(names)[S.argmax(1)]==k))
    return a
def run(args):
    step,n,c2,mode,seed,NC=args; rng=np.random.default_rng(seed+G.SEED_OFFSET); xg=G.make_grid(step)
    pool={s:{h:est(h,NC,c2,n,xg,s,mode,rng) for h in H3} for s in SURF}          # mau hieu chinh
    test={s:{h:est(h,NC,c2,n,xg,s,mode,rng) for h in H3} for s in SURF}          # mau kiem doc lap
    single={h:qfit(pool[ASSUMED][h]) for h in H3}
    mix={h:qfit(np.concatenate([pool[s][h] for s in TRAIN_MIX])) for h in H3}
    res={"single":{}, "mixture":{}, "oracle":{}}
    for s in SURF:
        res["single"][s]=acc(test[s],single)
        res["oracle"][s]=acc(test[s],{h:qfit(pool[s][h]) for h in H3})   # biet dung be mat (tran tren)
        if s not in TRAIN_MIX: res["mixture"][s]=acc(test[s],mix)
    return (step,n,c2,mode,res)
if __name__=="__main__":
    scale=float(sys.argv[1]) if len(sys.argv)>1 else 1.0; NC=int(1500*scale); jobs=[]; k=0
    for step,n in [(2.0,5),(2**.5,5),(2**.5,8)]:
        for c2 in [0.047,0.1,0.2,0.5]:
            for mode in ["naive","mask"]:
                jobs.append((step,n,c2,mode,31000+k,NC)); k+=1
    with Pool(G.NPROC) as p: out=p.map(run,jobs)
    print("Cot: acc3 trung binh / TOI TE NHAT tren cac be mat kiem (worst-case). Muc tieu >=0.9 o MOI be mat.")
    print("single = hieu chinh tren 1 be mat gia dinh (rho=.5,shift=0); mixture = hieu chinh tren 4 be mat, kiem tren 5 be mat con lai; oracle = biet dung be mat\n")
    print("step  n  c2     mode  | single mean/worst | mixture mean/worst | oracle mean/worst | be mat te nhat (single)")
    for step,n,c2,mode,r in out:
        f=lambda d:(np.mean([np.mean(list(v.values())) for v in d.values()]),min(np.mean(list(v.values())) for v in d.values()))
        sm,sw=f(r["single"]); mm,mw=f(r["mixture"]); om,ow=f(r["oracle"])
        worst=min(r["single"],key=lambda s:np.mean(list(r["single"][s].values())))
        flag="PASS" if mw>=0.9 else "FAIL"
        print(f"{step:<5.2f}{n:<3}{c2:<7}{mode:<6}| {sm:.3f} / {sw:.3f}     | {mm:.3f} / {mw:.3f}  {flag} | {om:.3f} / {ow:.3f}     | rho,shift={worst}")
