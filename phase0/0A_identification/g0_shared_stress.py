"""g0_shared_stress.py - Pha hong gia dinh cua estimator chia se: do cong thay doi theo rank (hinh dang KHONG chung).
c2(r)=c2*(r/8)^gamma, gamma in {-0.3,0,+0.3}. Hieu chinh QDA tren gamma=0, kiem tren gamma khac (be mat rho=.5, shift=0)."""
import numpy as np, sys, g0_final as G, g0_robustness as R, g0_shared as S
from multiprocessing import Pool
RANKS=G.RANKS
def batch(a,b,N,c2,n,xg,gamma,rng,rho=0.5,shift=0.0,cliff=5.0):
    xs=G.XREF-(a-b*G.P[:,None])*G.L[None,:]; d=xg[None,None,:]-xs[...,None]
    c2r=(c2*(RANKS/8.0)**gamma)[None,:,None]
    base=np.where(d<0,rho*c2r*d**2,c2r*d**2)
    col=xg[None,:]>np.log10(1e-3)-0.5*G.L[:,None]+shift
    base=base+cliff*col[None]
    return base[None]+rng.normal(0,R.SIG/np.sqrt(n),(N,)+base.shape),col
def est(h,N,c2,n,xg,gamma,mode,rng):
    y,col=batch(*R.H3[h],N,c2,n,xg,gamma,rng); W=np.broadcast_to((~col).astype(float),y.shape)
    e=R.vertex(y,W,xg); e=np.swapaxes(e,-1,-2).reshape(e.shape[:-2]+(-1,)); co=e@G.X_PINV.T
    if mode=="shared": th=S.refine(y,W,xg,co.copy()); return np.stack([-th[:,1],th[:,2]],-1)
    return np.stack([-co[:,1],co[:,2]],-1)
def run(args):
    step,n,c2,mode,seed,NC=args; rng=np.random.default_rng(seed+G.SEED_OFFSET); xg=G.make_grid(step)
    cal={h:R.qfit(est(h,NC,c2,n,xg,0.0,mode,rng)) for h in R.H3}; out={}
    for gm in [-0.3,0.0,0.3]:
        t={h:est(h,NC,c2,n,xg,gm,mode,rng) for h in R.H3}
        out[gm]=(float(np.mean(list(R.acc(t,cal).values()))), np.round(np.median(t["k1"],0)-np.array(R.H3["k1"]),3).tolist())
    return step,n,c2,mode,out
if __name__=="__main__":
    scale=float(sys.argv[1]) if len(sys.argv)>1 else 1.0; NC=int(800*scale); jobs=[];k=0
    for step,n in [(2.0,5),(2**.5,5),(2**.5,8)]:
        for c2 in [0.047,0.1,0.2,0.5]:
            for mode in ["mask","shared"]:
                jobs.append((step,n,c2,mode,63000+k,NC)); k+=1
    with Pool(G.NPROC) as p: res=p.map(run,jobs)
    print("Hieu chinh QDA tren gamma=0, kiem tren gamma=-0.3 / 0 / +0.3 (do cong thay doi theo rank). acc3 (bias k1 a,b)\n")
    print("step  n  c2     mode   | gamma=-0.3            | gamma=0               | gamma=+0.3")
    for step,n,c2,mode,o in res:
        f=lambda g:f"{o[g][0]:.3f} {tuple(o[g][1])}"
        print(f"{step:<5.2f}{n:<3}{c2:<7}{mode:<7}| {f(-0.3):<22}| {f(0.0):<22}| {f(0.3)}")
