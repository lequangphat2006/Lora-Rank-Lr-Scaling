"""g0_protocol.py - GIAO THUC G0 DAY DU: phan loai bang estimator chia se + chan doan DONG THUAN voi estimator mask.
Y tuong: khi hinh dang be mat dung (chia se giua cac rank), shared va mask cho ket qua khop theo mot cach co the hieu chuan (mask co do lech co dinh);
khi sai (vd do cong doi theo rank, gamma != 0) hai estimator tach nhau => CO CHAN ('flag'). Do:
  acc_all      : do chinh xac phan loai 3 lop (khong dung co)
  flag%        : ty le bi co chan (khong ket luan)
  unsafe%      : P(khong bi chan VA phan loai sai)  <- cai thuc su gay hai; muc tieu <= 5% o MOI be mat, MOI gamma
  acc|unflag   : chinh xac khi duoc phep ket luan; muc tieu >= 0.9 tai gamma=0
Hieu chuan: QDA ben (cat tia) tren 4 be mat, gamma=0; nguong co = phan vi 95% cua D tren 5 be mat con lai, gamma=0 (tuc FPR ~5% theo thiet ke).
Kiem tra: gamma in {-.3,-.15,0,.15,.3} tren 9 be mat (gamma=0: 5 be mat chua dung de hieu chuan)."""
import numpy as np, sys, g0_final as G, g0_robustness as R, g0_shared as S, g0_shared_stress as T
from multiprocessing import Pool
GAM=[-0.3,-0.15,0.0,0.15,0.3]
def both(h,N,c2,n,xg,gamma,surf,rng):
    y,col=T.batch(*R.H3[h],N,c2,n,xg,gamma,rng,rho=surf[0],shift=surf[1])
    W=np.broadcast_to((~col).astype(float),y.shape)
    e=R.vertex(y,W,xg); e=np.swapaxes(e,-1,-2).reshape(e.shape[:-2]+(-1,)); co=e@G.X_PINV.T
    th=S.refine(y,W,xg,co.copy())
    return np.column_stack([-th[:,1],th[:,2],-co[:,1],co[:,2]])           # shared(a,b), mask(a,b)
def maha(X,m,Ci): d=X-m; return np.einsum('ni,ij,nj->n',d,Ci,d)
def run(args):
    step,n,c2,seed,NC,NT,q=args; rng=np.random.default_rng(seed+G.SEED_OFFSET); xg=G.make_grid(step); H=list(R.H3)
    cal={h:np.concatenate([both(h,NC,c2,n,xg,0.0,s,rng) for s in R.TRAIN_MIX]) for h in H}
    mod={}
    for h in H:
        m,C=R.rfit(cal[h]); mod[h]=dict(m=m,Ci=np.linalg.inv(C),ld=np.log(np.linalg.det(C)),m2=m[:2],C2i=np.linalg.inv(C[:2,:2]),ld2=np.log(np.linalg.det(C[:2,:2])))
    def classify(X):
        sc=np.stack([-0.5*maha(X[:,:2],mod[k]["m2"],mod[k]["C2i"])-0.5*mod[k]["ld2"] for k in H],1); return sc.argmax(1)
    def D_of(X,pred): return np.array([maha(X[i:i+1],mod[H[pred[i]]]["m"],mod[H[pred[i]]]["Ci"])[0] for i in range(len(X))]) if len(X)<0 else \
        np.select([pred==j for j in range(3)],[maha(X,mod[H[j]]["m"],mod[H[j]]["Ci"]) for j in range(3)])
    held=[s for s in R.SURF if s not in R.TRAIN_MIX]
    nullD=np.concatenate([D_of(X:=both(h,NC,c2,n,xg,0.0,s,rng),np.full(len(X),H.index(h))) for h in H for s in held])
    thr=float(np.quantile(nullD,q)); res={}
    for g in GAM:
        surfs=held if g==0.0 else R.SURF; per=[]
        for s in surfs:
            acc=[];fl=[];uns=[];accu=[]
            for h in H:
                X=both(h,NT,c2,n,xg,g,s,rng); pred=classify(X); D=D_of(X,pred); ok=pred==H.index(h); f=D>thr
                acc.append(ok.mean()); fl.append(f.mean()); uns.append(((~f)&(~ok)).mean()); accu.append(ok[~f].mean() if (~f).any() else np.nan)
            per.append((np.mean(acc),np.mean(fl),np.mean(uns),np.nanmean(accu)))
        per=np.array(per); res[g]=dict(acc=per[:,0].mean(),acc_min=per[:,0].min(),flag=per[:,1].mean(),unsafe=per[:,2].mean(),unsafe_max=per[:,2].max(),accu=np.nanmean(per[:,3]),accu_min=np.nanmin(per[:,3]))
    return step,n,c2,thr,res
if __name__=="__main__":
    scale=float(sys.argv[1]) if len(sys.argv)>1 else 1.0; NC=int(600*scale); NT=int(300*scale); jobs=[];k=0
    for step,n in [(2.0,5),(2**.5,5),(2**.5,8)]:
        for c2 in [0.047,0.1,0.2,0.5]:
            jobs.append((step,n,c2,81000+k,NC,NT,0.95)); k+=1
    with Pool(G.NPROC) as p: out=p.map(run,jobs)
    print(__doc__.split("Hieu chuan")[0]); print("Moi o: acc_all(min bs) | flag% | unsafe%(max bs) | acc|unflag(min bs)\n")
    for step,n,c2,thr,r in out:
        print(f"step={step:.2f} n={n} c2={c2}  (nguong co D>{thr:.1f})")
        for g in GAM:
            x=r[g]; tag="" 
            if g==0.0: tag=" PASS" if (x['accu_min']>=0.9 and x['flag']<=0.15) else " FAIL"
            else: tag=" an toan" if x['unsafe_max']<=0.05 else " KHONG an toan"
            print(f"   gamma={g:+.2f} | acc {x['acc']:.3f} ({x['acc_min']:.3f}) | flag {100*x['flag']:5.1f}% | unsafe {100*x['unsafe']:5.1f}% ({100*x['unsafe_max']:5.1f}%) | acc|unflag {x['accu']:.3f} ({x['accu_min']:.3f}){tag}")
        print()
