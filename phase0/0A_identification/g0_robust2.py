"""Tach: (i) 3 lop G0 vs 5 lop, (ii) uoc luong naive vs loai diem collapse, (iii) do lon vach. Mo hinh bề mặt = script nguoi dung (plateau trai = 0.5*c2, vach collapse phu thuoc rank)."""
import numpy as np, g0_final as G, sys
from multiprocessing import Pool
P=G.P; L=G.L; XREF=G.XREF
H5={"k1":(1,1),"muA":(.5,.5),"khalf":(.5,1),"i1":(.82,1),"i2":(.8,.6)}
XG=np.log10(2e-5)+np.log10(2.0)*np.arange(7)
def batch(a,b,N,c2,sig,n,cliff,rng):
    xs=XREF-(a-b*P[:,None])*L[None,:]                       # (p,r)
    d=XG[None,None,:]-xs[...,None]
    base=np.where(d<0,0.5*c2*d**2,c2*d**2)
    col=XG[None,:]>np.log10(1e-3)-0.5*L[:,None]              # (r,m)
    base=base+cliff*col[None]
    return base[None]+rng.normal(0,sig/np.sqrt(n),(N,)+base.shape), col
def vertex(y,w):
    xc=XG-XG.mean(); V=np.vander(xc,3)
    w=np.broadcast_to(w,y.shape).astype(float)
    A=np.einsum('mi,...m,mj->...ij',V,w,V)+1e-12*np.eye(3)
    r=np.einsum('mi,...m->...i',V,w*y)
    c=np.linalg.solve(A,r[...,None])[...,0]
    ok=c[...,0]>0; safe=np.where(ok,c[...,0],1)
    fb=XG[np.argmin(np.where(w>0,y,np.inf),axis=-1)]-XG.mean()
    return np.where(ok,-c[...,1]/(2*safe),fb)+XG.mean()
def ab(y,w):
    e=vertex(y,w); e=np.swapaxes(e,-1,-2).reshape(e.shape[:-2]+(-1,)); co=e@G.X_PINV.T
    return np.stack([-co[...,1],co[...,2]],-1)
def est(a,b,N,c2,sig,n,cliff,mode,rng):
    y,col=batch(a,b,N,c2,sig,n,cliff,rng)
    w=np.ones(col.shape) if mode=="naive" else (~col).astype(float)
    return ab(y,w)
import g0_robustness as _R
def qda_fit(pts):   # v3.2: uoc luong cat tia ben voi ngoai le (cung bo phan loai voi buoc 08-11)
    return _R.qfit(pts)
def score(pt,q): m,Ci,ld=q; d=pt-m; return -0.5*np.einsum('ni,ij,nj->n',d,Ci,d)-0.5*ld
def run(args):
    c2,n,cliff,mode,seed=args; rng=np.random.default_rng(seed+G.SEED_OFFSET); sig=0.0136
    cal={k:qda_fit(est(*v,3000,c2,sig,n,cliff,mode,rng)) for k,v in H5.items()}
    tst={k:est(*v,2000,c2,sig,n,cliff,mode,rng) for k,v in H5.items()}
    out={}
    for lab,names in [("3cls",["k1","muA","khalf"]),("5cls",list(H5))]:
        acc={}
        for k in names:
            S=np.stack([score(tst[k],cal[j]) for j in names],1); acc[k]=float(np.mean(np.array(names)[S.argmax(1)]==k))
        out[lab]=(float(np.mean(list(acc.values()))),acc)
    bias={k:(np.median(tst[k],0)-np.array(H5[k])).round(3).tolist() for k in H5}
    return (c2,n,cliff,mode,out,bias)
if __name__=="__main__":
    jobs=[];s=0
    for cliff in [5.0,0.5]:
        for mode in ["naive","mask"]:
            for n in [3,5]:
                for c2 in [0.047,0.1,0.2,0.5]:
                    jobs.append((c2,n,cliff,mode,7000+s)); s+=1
    with Pool(G.NPROC) as p: R=p.map(run,jobs)
    print("cliff mode  seeds c2    | acc3(G0)  acc5   | worst-class3 | bias (a,b) cua k1 / i1")
    for c2,n,cliff,mode,o,b in R:
        w3=min(o["3cls"][1].values())
        print(f"{cliff:<5}{mode:<6}{n:<6}{c2:<6}| {o['3cls'][0]:.3f}    {o['5cls'][0]:.3f}  | {w3:.3f}        | k1{b['k1']} i1{b['i1']}")
