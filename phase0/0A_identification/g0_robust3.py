import numpy as np, g0_robust2 as R, g0_final as G
from multiprocessing import Pool
def run(args):
    step,c2,n,mode,seed=args; R.XG=G.make_grid(step)
    return (step,len(R.XG))+R.run((c2,n,5.0,mode,seed))
if __name__=="__main__":
    jobs=[];s=0
    for step in [2.0,2**.5,1.25]:
        for mode in ["naive","mask"]:
            for n in [5,8]:
                for c2 in [0.047,0.1,0.2]:
                    jobs.append((step,c2,n,mode,9000+s)); s+=1
    with Pool(G.NPROC) as p: out=p.map(run,jobs)
    print("step  pts mode  seeds c2   | acc3(G0)  acc5  | worst3 | bias k1(a,b)")
    for st,pts,c2,n,cl,mode,o,b in out:
        print(f"{st:<5.2f}{pts:<4}{mode:<6}{n:<6}{c2:<6}| {o['3cls'][0]:.3f}    {o['5cls'][0]:.3f} | {min(o['3cls'][1].values()):.3f}  | {b['k1']}")
