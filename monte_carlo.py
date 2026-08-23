#!/usr/bin/env python3
"""
monte_carlo.py — direct stochastic validation, independent of the eigenproblem.

Scaled units: dy = (s + y^2) dtau + sqrt(2) dW,  ramp ds/dtau = SIG = eps/(D/2).
Conditioning is on non-explosion (paths killed on exceeding Y_ESC).

This version incorporates an external audit.  Changes from the first draft:
  * error bars from INDEPENDENT BATCHES (the previous split-half estimator was a
    1-dof statistic on correlated halves; it spanned 3 orders of magnitude across
    statistically identical runs and produced a spurious z=7.2)
  * Y_ESC study by common random numbers (one ensemble, first-passage to every
    threshold) so it can actually detect a dependence
  * lambda tabulated on ds=0.02 for the hazard integral (trapezoid on convex
    lambda biased H high, same sign as the effect under test)
  * QSD normalisation includes the analytic tail mass beyond yR
  * Test 4 reports Delta = log(P_MC/P_pred) scaled by lambda, which is the
    invariant form; three ramp rates
  * conclusions are ASSERTED, not printed unconditionally

Runtime ~25 min.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SQ2 = np.sqrt(2.0)
SEEDS = np.random.SeedSequence(20260811).spawn(6)
R1, R2, R3, R4, R5, R6 = [np.random.default_rng(s) for s in SEEDS]
FAIL = []


def hdr(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78, flush=True)


def check(name, cond, detail):
    tag = "PASS" if cond else "**FAIL**"
    if not cond:
        FAIL.append(name)
    print(f"  [{tag}] {name}: {detail}", flush=True)


# ------------------------------------------------------------------ reference
def qsd(s, yR=12.0, L=9.0, N=150000, D=2.0, k=1):
    """Conditioned law and decay rate.  Normalisation includes the analytic
    flux tail beyond yR, int_yR^inf lambda/(s+y^2) dy, which the truncated
    integral omits (~lambda/yR, i.e. a few per cent)."""
    y = np.linspace(-L, yR, N + 1)
    h = y[1] - y[0]
    yi = y[1:-1]
    a = D / 2.0
    lo = a / h**2 * np.ones(len(yi))
    di = -2 * a / h**2 * np.ones(len(yi))
    up = a / h**2 * np.ones(len(yi))
    lo = lo + (s + np.roll(yi, 1)**2) / (2 * h)
    up = up - (s + np.roll(yi, -1)**2) / (2 * h)
    A = sp.diags([lo[1:], di, up[:-1]], [-1, 0, 1], format='csc')
    v0 = np.exp(-0.5 * (yi + np.sqrt(max(-s, 0.0)))**2); v0 /= np.linalg.norm(v0)
    vals, vecs = spla.eigs(A, k=k, sigma=0.0, which='LM', maxiter=10000, v0=v0)
    idx = np.argsort(-np.real(vals))          # least negative first
    lam = -np.real(vals[idx])
    rho = np.real(vecs[:, idx[0]])
    if rho.sum() < 0:
        rho = -rho
    rho = np.clip(rho, 0, None)
    Zb = np.trapezoid(rho, yi)
    b = np.sqrt(-s)
    tail = lam[0] * np.log((yR + b) / (yR - b)) / (2 * b)   # int_yR^inf lam/(y^2-|s|)
    rho = rho / (Zb * (1.0 + tail))
    return (yi, rho, float(lam[0])) if k == 1 else (yi, rho, lam)


# --------------------------------------------------------------- simulation
def run_batches(s, nb, npb, tmax, dtau, y_esc, rng, record=None):
    """nb independent batches of npb paths at fixed s.
    Returns times, survival counts (nb, nsteps+1), and a survivor snapshot."""
    nst = int(round(tmax / dtau))
    rec_k = None if record is None else int(round(record / dtau))
    sq = SQ2 * np.sqrt(dtau)
    ts = np.arange(nst + 1) * dtau
    na = np.zeros((nb, nst + 1))
    na[:, 0] = npb
    y = np.full(nb * npb, -np.sqrt(-s))
    lab = np.repeat(np.arange(nb), npb)
    snap = None
    nonfinite = 0
    for k in range(1, nst + 1):
        if y.size:
            y = y + (s + y * y) * dtau + sq * rng.standard_normal(y.size)
            bad = ~np.isfinite(y)
            nonfinite += int(bad.sum())
            keep = np.isfinite(y) & (y <= y_esc)
            y, lab = y[keep], lab[keep]
        na[:, k] = np.bincount(lab, minlength=nb) if y.size else 0
        if rec_k is not None and k == rec_k:
            snap = y.copy()
    return ts, na, snap, nonfinite


def lam_batches(ts, na, t_lo, t_hi):
    """Per-batch log-linear fit; report mean and standard error of the mean."""
    m = (ts >= t_lo) & (ts <= t_hi)
    out = []
    for row in na:
        mm = m & (row > 30)
        if mm.sum() > 20:
            out.append(-np.polyfit(ts[mm], np.log(row[mm]), 1)[0])
    out = np.array(out)
    return out.mean(), out.std(ddof=1) / np.sqrt(len(out)), len(out)


# ================================================================== TEST 1
hdr("TEST 1  Simulation convergence (independent-batch error bars)")
S1 = -0.5
_, _, LAM1 = qsd(S1)
print(f"  reference eigenvalue at s={S1}: lambda = {LAM1:.6f}\n")
print("  (a) time step   (Y_ESC=10, 16 batches x 2500)")
print("  %10s %13s %11s %9s" % ("dtau", "lambda_MC", "sem", "z"))
zs = []
for dtau in [4e-3, 2e-3, 1e-3, 5e-4]:
    ts, na, _, nf = run_batches(S1, 16, 2500, 15.0, dtau, 10.0, R1)
    lam, sem, nb = lam_batches(ts, na, 4.0, 15.0)
    z = (lam - LAM1) / sem
    zs.append(z)
    print("  %10.0e %13.6f %11.6f %9.2f" % (dtau, lam, sem, z))
check("no dtau trend", max(abs(np.array(zs))) < 3.0,
      f"max |z| over dtau = {max(abs(np.array(zs))):.2f} (want < 3)")

print("\n  (b) escape threshold, COMMON RANDOM NUMBERS")
print("      one ensemble; a path escapes at Y the first time it exceeds Y,")
print("      so all thresholds share the same paths (near-perfect correlation).")
THR = np.array([5.0, 7.0, 10.0, 13.0])
nb, npb, dtau, tmax = 16, 2500, 1e-3, 15.0
nst = int(tmax / dtau)
y = np.full(nb * npb, -np.sqrt(-S1))
lab = np.repeat(np.arange(nb), npb)
tcross = np.full((len(THR), nb * npb), np.inf)
idx_all = np.arange(nb * npb)
sq = SQ2 * np.sqrt(dtau)
alive_idx = idx_all.copy()
for k in range(1, nst + 1):
    if y.size:
        y = y + (S1 + y * y) * dtau + sq * R2.standard_normal(y.size)
        for j, Y in enumerate(THR):
            hit = y > Y
            if hit.any():
                gi = alive_idx[hit]
                new = np.isinf(tcross[j, gi])
                tcross[j, gi[new]] = k * dtau
        keep = y <= THR.max()
        y, alive_idx = y[keep], alive_idx[keep]
ts = np.arange(nst + 1) * dtau
print("  %10s %13s %11s %9s" % ("Y_ESC", "lambda_MC", "sem", "z"))
zs = []
for j, Y in enumerate(THR):
    na = np.zeros((nb, nst + 1))
    for b in range(nb):
        sel = (lab == b)
        tc = tcross[j, sel]
        na[b] = (tc[None, :] > ts[:, None]).sum(axis=1)
    lam, sem, _ = lam_batches(ts, na, 4.0, 15.0)
    z = (lam - LAM1) / sem
    zs.append(z)
    print("  %10.1f %13.6f %11.6f %9.2f" % (Y, lam, sem, z))
check("no Y_ESC trend", abs(zs[-1] - zs[0]) < 4.0,
      f"z spread across thresholds = {abs(zs[-1]-zs[0]):.2f} (want < 4)")

# ================================================================== TEST 2
hdr("TEST 2  Escape rate lambda(s): Monte Carlo vs eigenvalue")
print("  %8s %13s %13s %11s %8s" % ("s", "lambda_MC", "lambda_eig", "sem", "z"))
zs = []
for s, tmax, dtau, lo, hi in [(-0.25, 12.0, 1e-3, 3.0, 12.0),
                              (-0.50, 20.0, 1e-3, 4.0, 20.0),
                              (-1.00, 55.0, 1e-3, 12.0, 55.0),
                              (-2.00, 300.0, 2e-3, 60.0, 300.0)]:
    _, _, le = qsd(s)
    ts, na, _, _ = run_batches(s, 16, 3000, tmax, dtau, 10.0, R3)
    lam, sem, _ = lam_batches(ts, na, lo, hi)
    z = (lam - le) / sem
    zs.append(abs(z))
    print("  %8.2f %13.6f %13.6f %11.6f %8.2f" % (s, lam, le, sem, z))
check("lambda matches eigenvalue", max(zs) < 3.0,
      f"max |z| = {max(zs):.2f} (want < 3)")

# ================================================================== TEST 3
hdr("TEST 3  Conditioned density from survivors vs eigenvector")
s3 = -0.5
yi, rho, lam3 = qsd(s3, yR=40.0, N=200000)
ts, na, snap, nf = run_batches(s3, 1, 150000, 22.0, 1e-3, 40.0, R4, record=14.0)
print(f"  survivors at tau=14: {snap.size};  non-finite states: {nf}")
edges = np.linspace(-3.0, 7.0, 70)
cent = 0.5 * (edges[1:] + edges[:-1]); w = np.diff(edges)
cnt, _ = np.histogram(snap, bins=edges)
pdf_raw = cnt / (cnt.sum() * w)
err_raw = np.sqrt(np.maximum(cnt, 1)) / (cnt.sum() * w)
Z = np.trapezoid(pdf_raw, cent)
pdf, err = pdf_raw / Z, err_raw / Z
ref = np.interp(cent, yi, rho); ref = ref / np.trapezoid(ref, cent)
sel = cnt > 25
chi2 = np.sum(((pdf[sel] - ref[sel]) / err[sel])**2) / (sel.sum() - 1)
print(f"  reduced chi^2 = {chi2:.2f} over {sel.sum()} bins (dof = n-1)")
check("density matches eigenvector", 0.5 < chi2 < 1.8, f"chi^2/dof = {chi2:.2f}")
print("\n  flux tail  rho(y)(s+y^2)/lambda -> 1   (yR=40, tail-corrected norm)")
print("  %8s %12s %12s %10s" % ("y", "MC", "eigen", "MC sem"))
for yt in [2.0, 3.0, 4.0, 5.0]:
    i = int(np.argmin(np.abs(cent - yt)))
    if cnt[i] < 10:
        continue
    f = (s3 + cent[i]**2) / lam3
    print("  %8.1f %12.4f %12.4f %10.4f" % (cent[i], pdf[i]*f, ref[i]*f, err[i]*f))

# ================================================================== TEST 4
hdr("TEST 4  Ramped survival: adiabatic lag correction")
Sg = -np.arange(6.0, 0.28, -0.02)                       # fine grid (audit B3)
Lam = np.array([qsd(sv, yR=8.0, N=80000)[2] for sv in Sg])
check("lambda tabulation positive/monotone", (Lam > 0).all() and
      (np.diff(Lam) > -1e-12).all(), f"{len(Sg)} points, min={Lam.min():.2e}")
_, _, l12 = qsd(-0.5, k=3)
gap = float(l12[1] - l12[0])
print(f"  spectral gap at s=-0.5: {gap:.4f}  ->  1/gap = {1/gap:.4f}")

results = {}
for SIG in [0.05, 0.02, 0.005]:
    nb, npb, dtau, s0, s_end = 8, 6000, 1e-3, -6.0, -0.30
    n = nb * npb
    y = np.full(n, -np.sqrt(-s0)); lab = np.repeat(np.arange(nb), npb)
    s_cur = s0; sq = SQ2 * np.sqrt(dtau)
    rec_s, rec_P = [], []
    next_rec = -5.0
    while s_cur < s_end:
        if y.size:
            y = y + (s_cur + y * y) * dtau + sq * R5.standard_normal(y.size)
            keep = np.isfinite(y) & (y <= 10.0)
            y, lab = y[keep], lab[keep]
        s_cur += SIG * dtau
        if s_cur >= next_rec:
            cb = np.bincount(lab, minlength=nb) if y.size else np.zeros(nb)
            rec_s.append(s_cur); rec_P.append(cb / npb)
            next_rec += 0.10
    rec_s = np.array(rec_s); rec_P = np.array(rec_P)      # (nrec, nb)
    Pm = rec_P.mean(axis=1); Pse = rec_P.std(axis=1, ddof=1) / np.sqrt(nb)
    Hp = np.concatenate([[0.0],
                         np.cumsum(0.5*(Lam[1:]+Lam[:-1])*np.diff(Sg))/SIG])
    Ppred = np.exp(-np.interp(rec_s, Sg, Hp))             # interp H, not exp(-H)
    lam_at = np.interp(rec_s, Sg, Lam)
    ok = (Pm > 0.02) & (Pm < 0.98)
    D_ = np.log(Pm[ok] / Ppred[ok])
    ratio = D_ / lam_at[ok]
    results[SIG] = (abs(rec_s[ok]), D_, ratio, lam_at[ok])
    print(f"\n  SIG = {SIG}   (n={n} in {nb} batches)")
    print("  %7s %10s %10s %11s %11s" % ("|s|", "P_MC", "P_pred", "Delta", "Delta/lam"))
    for k in range(0, ok.sum(), max(1, ok.sum()//7)):
        print("  %7.2f %10.4f %10.4f %11.4f %11.3f"
              % (results[SIG][0][k], Pm[ok][k], Ppred[ok][k], D_[k], ratio[k]))
    print(f"    mean Delta/lambda = {ratio.mean():.3f} +/- {ratio.std(ddof=1)/np.sqrt(len(ratio)):.3f}")

taus = np.array([results[S][2].mean() for S in results])
print(f"\n  tau_eff across SIG = {np.array2string(taus, precision=3)}")
check("Delta/lambda is SIG-independent (adiabatic lag)",
      taus.std() / taus.mean() < 0.12,
      f"spread {100*taus.std()/taus.mean():.1f}% of mean {taus.mean():.3f}")
check("tau_eff is of order 1/gap", 0.5 < taus.mean()*gap < 3.0,
      f"tau_eff*gap = {taus.mean()*gap:.2f}")

hdr("SUMMARY")
if FAIL:
    print("  FAILED CHECKS: " + ", ".join(FAIL))
else:
    print("  All checks passed.")
