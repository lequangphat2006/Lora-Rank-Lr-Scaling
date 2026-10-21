"""Chay lai logic power goc (khong sua) + doi chung am + in do rong CI/SE."""
import numpy as np
from scipy.stats import linregress
ranks=np.array([1,2,4,8,16,32,64]); x_grid=np.log10(np.logspace(np.log10(2e-5),np.log10(2e-3),9)); x_ref=np.log10(5e-4); r_ref=8
def run(c2,n_seeds,noise,n_iter=500,per_seed_noise=True):
    eff = noise/np.sqrt(n_seeds)
    def surf(a,b,p):
        xs=x_ref-(a-b*p)*np.log10(ranks/r_ref)
        return np.array([c2*(x_grid-xs[i])**2+np.random.normal(0,eff,9) for i in range(7)])
    def est(s):
        e=[]
        for i in range(7):
            c=np.polyfit(x_grid,s[i],2); e.append(-c[1]/(2*c[0]) if c[0]>0 else x_grid[np.argmin(s[i])])
        r=linregress(np.log10(ranks),e); return -r.slope,r.stderr,np.array(e)
    p1=p2=0; ses=[]; sl=[]; edge=0
    for _ in range(n_iter):
        s,se,e=est(surf(1,1,0)); ses.append(se); sl.append(s)
        if s-2.57*se>0.5 or s+2.57*se<0.5: p1+=1
        edge+=np.mean((e<x_grid[0])|(e>x_grid[-1]))
        s,se,_=est(surf(.5,.5,0))
        if s-2.57*se>1 or s+2.57*se<1: p2+=1
    return p1/n_iter,p2/n_iter,np.median(ses),np.std(sl),edge/n_iter
import os
np.random.seed(0+int(os.environ.get('G0_SEED_OFFSET','0')))
print("c2      noise     seeds | PowerH1 PowerMuA | median stderr(slope)  sd(slope across sims)  frac eta* outside grid")
for c2,noise,lab in [(0.047,0.0136,"goc"),(0.5,0.0136,"goc c2=.5"),(0.047,0.136,"noise x10"),(0.0005,0.0136,"c2~0"),(0.047,0.0136*30,"noise x30")]:
    for n in [3,8]:
        a,b,se,sd,ed=run(c2,n,noise); print(f"{c2:<7}{noise:<9.4f} {n:<5} | {a:6.2f} {b:8.2f} | {se:12.3f} {sd:20.3f} {ed:14.2f}   [{lab}]")
print("\nTinh ky vong ly thuyet: SD(slope) voi noise 0.0136, 3 seed, c2=.047 ~ tu mo phong tren")
