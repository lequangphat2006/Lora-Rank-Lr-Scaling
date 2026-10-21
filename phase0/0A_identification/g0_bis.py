"""g0_bis.py - 0A-bis: regret cua R-known, R-muA, R-est (R-probe hoan den 0B).

Regret(r,p) = c2 * (x_rule - x_true)^2 ,  x = log10(eta)
  x_rule - x_true = -[(a_rule-a_true) - (b_rule-b_true)*p] * L_r ,  L_r = log10(r/8)
R-known: oracle, regret = 0.
R-muA  : (a,b) = (0.5,0.5) co dinh.
R-est  : (a,b) uoc luong tu dung thiet ke 0A-bis (P1: 7 rank; P2,P3: 4 rank),
         hoi quy 2D tren cac o co du lieu. Giu tuong quan giua a_hat va b_hat.
Chi tieu: (i) regret trung binh tren cac cap (r,p) cua Phase 1, theo tung mau mo phong
          (ii) p95 cua (i)  (iii) ti le mau co cap (r,p) xau nhat > DELTA_R.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g0_final as G

SIGMA = 0.0136
DELTA_R = 0.02
N_MC = 400
P2P3_RANKS = [2, 8, 32, 64]
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs_bis")
os.makedirs(LOG_DIR, exist_ok=True)

ridx = {int(r): i for i, r in enumerate(G.RANKS)}
CELLS = [(0, ridx[int(r)]) for r in G.RANKS] + \
        [(pi, ridx[r]) for pi in (1, 2) for r in P2P3_RANKS]       # (p_index, rank_index)
PAIRS_L = np.array([G.L[ri] for _, ri in CELLS])
PAIRS_P = np.array([G.P[pi] for pi, _ in CELLS])
XS = np.array([[1.0, l, p * l] for l, p in zip(PAIRS_L, PAIRS_P)])
XS_PINV = np.linalg.pinv(XS)


def fit_ab_reduced(mean_surf, xg):
    """mean_surf (3,7,m) -> (a,b) chi dung cac o CELLS."""
    eta = G.vertex(mean_surf, xg)                      # (3,7)
    e = np.array([eta[pi, ri] for pi, ri in CELLS])
    co = XS_PINV @ e
    return -co[1], co[2]


def regret(a_rule, b_rule, a_true, b_true, c2):
    return c2 * ((a_rule - a_true) - (b_rule - b_true) * PAIRS_P) ** 2 * PAIRS_L ** 2


def summarize(per_sample):
    """per_sample: (N, n_pairs) -> mean(avg), p95(avg), frac worst>DELTA_R"""
    avg = per_sample.mean(1)
    worst = per_sample.max(1)
    return avg.mean(), np.percentile(avg, 95), np.mean(worst > DELTA_R)


def main():
    out = ["# 0A-bis: regret cua cac quy tac chuyen giao", ""]
    out.append(f"DELTA_R = {DELTA_R} (placeholder; chot trong R0). sigma = {SIGMA}. Mo phong {N_MC} mau / o.")
    out.append(f"Thiet ke: P1 = 7 rank {list(G.RANKS)}; P2,P3 = {P2P3_RANKS}. So cap (r,p) = {len(CELLS)}.")
    out.append("R-est: hoi quy 2D tren dung cac o tren (khong dung 7 rank o P2/P3). Luoi LR buoc 2 va sqrt2.")
    out.append("R-probe: KHONG tinh (can E_W tu 0B). Khong co bootstrap/phan loai o day: day la regret cua uoc luong diem.")
    out.append("Cot: mean = TB(regret trung binh theo cap); p95 = phan vi 95 cua no; %>d = ti le mau co cap xau nhat > DELTA_R.")
    out.append("")
    rng = np.random.default_rng(7 + G.SEED_OFFSET)
    for c2 in [0.047, 0.1, 0.2, 0.5]:
        for step, sname in [(2.0, "2"), (np.sqrt(2), "sqrt2")]:
            xg = G.make_grid(step)
            for n in [3, 5, 8]:
                out.append(f"## c2={c2} | buoc {sname} | {n} seed")
                out.append("| hyp | R-muA mean | R-muA %>d | R-est mean | R-est p95 | R-est %>d | PASS(R-est p95<=d) |")
                out.append("|---|---|---|---|---|---|---|")
                for h, (a, b) in G.H.items():
                    mu = regret(0.5, 0.5, a, b, c2)
                    mu_mean = mu.mean(); mu_frac = float(mu.max() > DELTA_R)
                    reg = np.empty((N_MC, len(CELLS)))
                    for k in range(N_MC):
                        d = G.simulate(a, b, xg, c2, SIGMA, n, rng)
                        ah, bh = fit_ab_reduced(d.mean(2), xg)
                        reg[k] = regret(ah, bh, a, b, c2)
                    m, p95, fr = summarize(reg)
                    out.append(f"| {h} | {mu_mean:.4f} | {'100%' if mu_frac else '0%'} | {m:.4f} | {p95:.4f} | {100*fr:.0f}% | {'PASS' if p95 <= DELTA_R else 'FAIL'} |")
                out.append("")
    text = "\n".join(out)
    path = os.path.join(LOG_DIR, "bis_01_regret.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("->", path)


if __name__ == "__main__":
    main()
