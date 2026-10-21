"""g0_final.py - G0 dung dinh nghia v3: P(phan loai dung)>=0.9 va coverage CI in [0.93,0.97].
Vector hoa. Thiet ke: buoc luoi x so seed. Co doi chung am. Seed bootstrap theo seed (B lan)."""
import numpy as np, sys, time, json, os
SEED_OFFSET=int(os.environ.get('G0_SEED_OFFSET','0'))   # doi hat giong de lap doc lap; 0 = nhu ban goc
NPROC=int(os.environ.get('G0_PROCS','2'))               # so tien trinh (Kaggle co 4 nhan: dat 4)
from multiprocessing import Pool

P = np.array([0.0, 0.5, 1.0]); RANKS = np.array([1,2,4,8,16,32,64]); R_REF = 8
XREF = np.log10(5e-4); L = np.log10(RANKS / R_REF)
H = {"k1": (1.0,1.0), "muA": (0.5,0.5), "khalf": (0.5,1.0)}
NAMES = list(H)

def make_grid(step_ratio):
    dx = np.log10(step_ratio); n = int(np.floor(2.0/dx + 1e-9)) + 1
    return np.log10(2e-5) + dx*np.arange(n)

# hoi quy 2D: x*(r,p) = b0 + b1*L + b2*p*L -> a=-b1, b=b2 (thu tu r-major, p-minor)
Xd = np.array([[1, l, p*l] for l in L for p in P]); X_PINV = np.linalg.pinv(Xd)

def vertex(y, xg):
    """y: (...,m) -> vertex parabol (...,), fallback argmin neu khong loi."""
    V = np.vander(xg, 3); Pi = np.linalg.pinv(V)
    c = y @ Pi.T
    ok = c[...,0] > 0
    safe = np.where(ok, c[...,0], 1.0)
    return np.where(ok, -c[...,1]/(2*safe), xg[np.argmin(y, axis=-1)])

def fit_ab(mean_surf, xg):
    """mean_surf: (...,n_p,n_r,m) -> a,b"""
    eta = vertex(mean_surf, xg)                      # (...,n_p,n_r)
    eta = np.swapaxes(eta, -1, -2).reshape(eta.shape[:-2] + (-1,))  # r-major
    co = eta @ X_PINV.T
    return -co[...,1], co[...,2]

def simulate(a, b, xg, c2, sigma, n, rng):
    xs = XREF - (a - b*P[:,None]) * L[None,:]        # (n_p,n_r)
    true = c2*(xg[None,None,:]-xs[...,None])**2      # (n_p,n_r,m)
    return true[:,:,None,:] + rng.normal(0, sigma, true.shape[:2]+(n,)+true.shape[2:])

def one(data, xg, B, rng):
    n = data.shape[2]
    a0, b0 = fit_ab(data.mean(2), xg)
    pi = np.arange(3)[None,:,None,None]; ri = np.arange(7)[None,None,:,None]
    idx = rng.integers(0, n, (B,3,7,n))
    boot = data[pi, ri, idx, :].mean(3)              # (B,3,7,m)
    ab, bb = fit_ab(boot, xg)
    return a0, b0, np.percentile(ab,[2.5,97.5]), np.percentile(bb,[2.5,97.5]), ab.std(), bb.std()

def run_design(args):
    step, n, c2, sigma, iters, B, seed, ncal = args
    rng = np.random.default_rng(seed+SEED_OFFSET); xg = make_grid(step)
    est = {h: [] for h in NAMES}; cov = {h: [0,0] for h in NAMES}; wid = {h: [[],[]] for h in NAMES}
    for h in NAMES:
        a, b = H[h]
        for _ in range(iters):
            d = simulate(a, b, xg, c2, sigma, n, rng)
            a0, b0, cia, cib, sa, sb = one(d, xg, B, rng)
            est[h].append((a0,b0))
            cov[h][0] += cia[0] <= a <= cia[1]; cov[h][1] += cib[0] <= b <= cib[1]
            wid[h][0].append(cia[1]-cia[0]); wid[h][1].append(cib[1]-cib[0])
    est = {h: np.array(v) for h,v in est.items()}
    # phan loai 1: khoang cach Euclid toi tam ly thuyet
    def eucl(pt): return NAMES[int(np.argmin([np.sum((pt-np.array(H[k]))**2) for k in NAMES]))]
    acc_e = {h: np.mean([eucl(p)==h for p in est[h]]) for h in NAMES}
    # phan loai 2: QDA, hieu chinh tren tap doc lap (ncal mau/gia thuyet)
    cal = {}
    for h in NAMES:
        a,b = H[h]
        pts = np.array([fit_ab(simulate(a,b,xg,c2,sigma,n,rng).mean(2), xg) for _ in range(ncal)])
        cal[h] = (pts.mean(0), np.cov(pts.T)+1e-9*np.eye(2))
    def qda(pt):
        s=[]
        for k in NAMES:
            m,C = cal[k]; dv = pt-m
            s.append(-0.5*dv@np.linalg.solve(C,dv)-0.5*np.log(np.linalg.det(C)))
        return NAMES[int(np.argmax(s))]
    acc_q = {h: np.mean([qda(p)==h for p in est[h]]) for h in NAMES}
    out = dict(step=float(step), n=n, c2=c2, sigma=sigma, grid_pts=len(xg),
        acc_eucl=float(np.mean(list(acc_e.values()))), acc_eucl_by=acc_e,
        acc_qda=float(np.mean(list(acc_q.values()))), acc_qda_by=acc_q,
        cov_a={h: cov[h][0]/iters for h in NAMES}, cov_b={h: cov[h][1]/iters for h in NAMES},
        width_a=float(np.mean([np.mean(wid[h][0]) for h in NAMES])), width_b=float(np.mean([np.mean(wid[h][1]) for h in NAMES])),
        bias_a={h: float(est[h][:,0].mean()-H[h][0]) for h in NAMES}, bias_b={h: float(est[h][:,1].mean()-H[h][1]) for h in NAMES},
        sd_a={h: float(est[h][:,0].std()) for h in NAMES}, sd_b={h: float(est[h][:,1].std()) for h in NAMES})
    out["cov_a_mean"]=float(np.mean(list(out["cov_a"].values()))); out["cov_b_mean"]=float(np.mean(list(out["cov_b"].values())))
    out["G0"] = bool(out["acc_qda"]>=0.9 and 0.93<=out["cov_a_mean"]<=0.97 and 0.93<=out["cov_b_mean"]<=0.97)
    return out

if __name__ == "__main__":
    mode = sys.argv[1]; iters=int(sys.argv[2]); B=int(sys.argv[3])
    S=0.0136
    if mode == "bench":
        t=time.time(); r=run_design((2.0,3,0.047,S,20,B,1,50)); print(time.time()-t, r["acc_eucl"]); sys.exit()
    jobs=[]; k=0
    if mode == "neg":   # doi chung am
        for c2,sig,lab in [(0.047,S,"audit"),(0.047,S*10,"noise x10"),(0.0005,S,"c2~0")]:
            jobs.append((2.0,3,c2,sig,iters,B,100+k,300)); k+=1
    else:               # quet luoi
        c2=float(sys.argv[4])
        for step in [2.5,2.0,np.sqrt(2)]:
            for n in [3,5,8]:
                jobs.append((step,n,c2,S,iters,B,1000+k,500)); k+=1
    with Pool(NPROC) as p: res=p.map(run_design, jobs)
    json.dump(res, open(f"res_{mode}_{sys.argv[4] if len(sys.argv)>4 else 'x'}.json","w"), indent=1)
    for r in res:
        print(f"step={r['step']:.2f} n={r['n']} c2={r['c2']} sig={r['sigma']:.4f} pts={r['grid_pts']} | accEucl={r['acc_eucl']:.3f} accQDA={r['acc_qda']:.3f} covA={r['cov_a_mean']:.3f} covB={r['cov_b_mean']:.3f} wA={r['width_a']:.2f} wB={r['width_b']:.2f} G0={r['G0']}")
