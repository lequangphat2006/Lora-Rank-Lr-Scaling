import numpy as np, json, sys
import g0_final as G
from multiprocessing import Pool
def one_p(data, xg, B, rng):
    n = data.shape[2]; mu = data.mean(2)
    a0, b0 = G.fit_ab(mu, xg)
    s2 = ((data-mu[:,:,None,:])**2).sum()/(3*7*len(xg)*(n-1)) if n>1 else 0.0136**2
    V = np.vander(xg,3); f = (mu @ np.linalg.pinv(V).T) @ V.T      # be mat parabol da fit
    boot = f[None] + rng.normal(0, np.sqrt(s2/n), (B,)+f.shape)
    ab, bb = G.fit_ab(boot, xg)
    return a0, b0, np.percentile(ab,[2.5,97.5]), np.percentile(bb,[2.5,97.5]), ab.std(), bb.std()
G.one = one_p
if __name__=="__main__":
    c2=float(sys.argv[1]); iters=int(sys.argv[2]); jobs=[]; k=0
    for step,n in [(1.778,1),(1.778,3),(2.5,3),(2.0,3),(2.0,5),(2.0,8),(2**.5,3),(2**.5,5),(2**.5,8)]:
        jobs.append((step,n,c2,0.0136,iters,500,5000+k,500)); k+=1
    with Pool(G.NPROC) as p: res=p.map(G.run_design, jobs)
    for r in res:
        print(f"[param-boot] step={r['step']:.2f} n={r['n']} pts={r['grid_pts']} | accEucl={r['acc_eucl']:.3f} accQDA={r['acc_qda']:.3f} covA={r['cov_a_mean']:.3f} covB={r['cov_b_mean']:.3f} wA={r['width_a']:.2f} wB={r['width_b']:.2f} biasA={np.mean(list(r['bias_a'].values())):+.3f} biasB={np.mean(list(r['bias_b'].values())):+.3f} sdA={np.mean(list(r['sd_a'].values())):.3f} sdB={np.mean(list(r['sd_b'].values())):.3f} G0={r['G0']}")
        print("     cov_a",{k:round(v,3) for k,v in r['cov_a'].items()},"cov_b",{k:round(v,3) for k,v in r['cov_b'].items()},"accQDA_by",{k:round(v,3) for k,v in r['acc_qda_by'].items()})
