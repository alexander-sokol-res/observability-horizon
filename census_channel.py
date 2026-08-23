#!/usr/bin/env python3
"""
The ensemble-tipping-fraction channel: how much of the tipping-time
information survives coarse-graining the tipping TIME down to a tipping
FRACTION?

Setting.  The ramped fold's tipping law is exactly unit-scale Gumbel in
w = (4/3) sigma^{3/2}: survival S = exp[-e^{-(w-mu)}], mu = ln(1/(2 pi SIG)).
Observing the exact tipping time gives F_tau -> 4 sigma_h, sigma_h=(3mu/4)^{2/3}.

Question: an observer who cannot time individual tippings, but can count what
FRACTION of an ensemble has tipped at m chosen parameter values, measures a
multinomial. How much information does that carry?

Verified here:
  (1) One binary snapshot per realization carries
          F_bin(sigma) = 4 sigma u^2/(e^u - 1),   u = e^{-(w-mu)},
      whose sigma-dependence is, to leading order, 4*sigma_h times a UNIVERSAL
      function of u alone.
  (2) Maximizing u^2/(e^u-1) gives u* = 1.59362, value 0.64761. A single
      yes/no observation recovers ~65% of the full timing channel, and
      inherits the same [ln(1/eps)]^{2/3} growth.
  (3) The optimum sits at survival S = e^{-u*} = 0.2032, i.e. observe when
      ~79.7% of the ensemble has tipped -- INDEPENDENT of SIG, D, ramp rate.
  (4) m snapshots nest: F_m increases monotonically to F_tau as m -> infinity.

Estimated parameter throughout: a pure location parameter theta in sigma=|s|
(the position of the bifurcation). Both channels are per realization, so the
comparison is like-for-like.
"""
import numpy as np
from scipy.optimize import brentq, minimize

rng = np.random.default_rng(20260815)


# ------------------------------------------------------------------ law
def w_of(sig):
    return (4.0 / 3.0) * sig ** 1.5


def sig_of(w):
    return (0.75 * w) ** (2.0 / 3.0)


def mu_of(SIG):
    return np.log(1.0 / (2.0 * np.pi * SIG))


def u_of(sig, mu):
    return np.exp(-(w_of(sig) - mu))


def S_of(sig, mu):
    return np.exp(-u_of(sig, mu))


# dS/dsigma = S * u * dw/dsigma,  dw/dsigma = 2 sqrt(sigma)
def dS_dsig(sig, mu):
    u = u_of(sig, mu)
    return np.exp(-u) * u * 2.0 * np.sqrt(sig)


# ------------------------------------------------------------------ (1) binary
def F_bin(sig, mu):
    """Fisher info of one tipped/not-tipped observation at sigma, about theta."""
    S = S_of(sig, mu)
    p = 1.0 - S
    d = dS_dsig(sig, mu)                      # dp/dsigma = -dS/dsigma
    return d * d / (p * (1.0 - p))


def F_bin_analytic(sig, mu):
    u = u_of(sig, mu)
    return 4.0 * sig * u * u / np.expm1(u)


# ------------------------------------------------------------------ full timing
def F_tau_mc(mu, M=4_000_000):
    """E_u[(2 sqrt(sig)(u-1) + 1/(2 sig))^2], sig=(3(mu-ln u)/4)^{2/3}."""
    u = rng.exponential(size=M)
    ok = np.log(u) < mu
    u = u[ok]
    sig = (0.75 * (mu - np.log(u))) ** (2.0 / 3.0)
    sc = 2.0 * np.sqrt(sig) * (u - 1.0) + 1.0 / (2.0 * sig)
    return float(np.mean(sc * sc))


# ------------------------------------------------------------------ (4) m snapshots
def F_multi(sigs, mu):
    """m snapshot parameter values (any order) -> multinomial Fisher info."""
    s = np.sort(np.asarray(sigs))[::-1]        # decreasing sigma = forward in time
    S = S_of(s, mu)
    dS = dS_dsig(s, mu)
    # bins: [tipped before s0], [s0..s1], ..., [alive at s_{m-1}]
    q = np.concatenate([[1.0 - S[0]], S[:-1] - S[1:], [S[-1]]])
    dq = np.concatenate([[-dS[0]], dS[:-1] - dS[1:], [dS[-1]]])
    good = q > 1e-300
    return float(np.sum(dq[good] ** 2 / q[good]))


def best_design(m, mu):
    """Optimal m-snapshot design, parameterized by u (log-spaced, unconstrained)."""
    def neg(z):
        sig = sig_of(mu - z)                   # z = ln u ; w = mu - z
        if np.any(sig <= 0) or np.any(~np.isfinite(sig)):
            return 1e9
        return -F_multi(sig, mu)
    best = None
    for _ in range(40):
        z0 = np.sort(rng.uniform(-3.0, 3.0, size=m))
        r = minimize(neg, z0, method='Nelder-Mead',
                     options=dict(maxiter=20000, xatol=1e-10, fatol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, np.sort(sig_of(mu - best.x))[::-1]


# ================================================================== run
print("=" * 78)
print("(0) universal constant: maximize g(u) = u^2/(e^u - 1)")
print("=" * 78)
gp = lambda u: 2.0 * np.expm1(u) - u * np.exp(u)      # d/du of u^2/(e^u-1), numerator
ustar = brentq(gp, 0.5, 5.0, xtol=1e-14)
gstar = ustar ** 2 / np.expm1(ustar)
print(f"  u*     = {ustar:.10f}")
print(f"  g(u*)  = {gstar:.10f}      <- efficiency of ONE binary snapshot")
print(f"  S(u*)  = {np.exp(-ustar):.10f}      <- survival at the optimum")
print(f"  tipped = {1 - np.exp(-ustar):.10f}      <- UNIVERSAL optimal tipping fraction")

print("\n" + "=" * 78)
print("(1) analytic F_bin vs numerical-derivative F_bin  (SIG=2.222e-3)")
print("=" * 78)
mu = mu_of(2.222e-3)
print(f"{'sigma':>9} {'F_bin (formula)':>17} {'F_bin (finite diff)':>21} "
      f"{'rel err':>11}")
for sig in [1.0, 2.0, sig_of(mu - np.log(ustar)), 4.0, 5.0]:
    h = 1e-6
    p = lambda x: 1.0 - S_of(x, mu)
    dp = (p(sig + h) - p(sig - h)) / (2 * h)
    fd = dp * dp / (p(sig) * (1 - p(sig)))
    fa = F_bin_analytic(sig, mu)
    print(f"{sig:9.4f} {fa:17.8f} {fd:21.8f} {abs(fa / fd - 1):11.2e}")

print("\n" + "=" * 78)
print("(2,3) one binary snapshot vs the full timing channel")
print("=" * 78)
print(f"{'SIG':>10} {'sigma_h':>9} {'F_tau(MC)':>10} {'F_bin max':>10} "
      f"{'sigma*':>9} {'tipped frac':>12} {'F_bin/F_tau':>12}")
SIGS = [2.222e-2, 2.222e-3, 2.222e-4, 2.222e-5, 2.222e-6]
rows = []
for SIG in SIGS:
    mu = mu_of(SIG)
    sh = sig_of(mu)
    # exact maximization of F_bin over sigma (not the leading-order form)
    z = np.linspace(-6, 6, 200001)
    sig = sig_of(np.clip(mu - z, 1e-12, None))
    F = F_bin_analytic(sig, mu)
    i = int(np.argmax(F))
    r = minimize(lambda a: -F_bin_analytic(np.clip(a[0], 1e-9, None), mu),
                 [sig[i]], method='Nelder-Mead',
                 options=dict(xatol=1e-12, fatol=1e-14))
    sstar = float(r.x[0])
    Fmax = -float(r.fun)
    Ft = F_tau_mc(mu)
    tf = 1.0 - S_of(sstar, mu)
    rows.append((SIG, sh, Ft, Fmax, sstar, tf, Fmax / Ft))
    print(f"{SIG:10.3e} {sh:9.4f} {Ft:10.4f} {Fmax:10.4f} {sstar:9.4f} "
          f"{tf:12.6f} {Fmax / Ft:12.4f}")
print(f"\n  predicted universal tipped fraction : {1 - np.exp(-ustar):.6f}")
print(f"  predicted efficiency in the deep limit: {gstar:.6f}")

print("\n" + "=" * 78)
print("(4) m snapshots nest upward to the full timing channel  (SIG=2.222e-3)")
print("=" * 78)
mu = mu_of(2.222e-3)
Ft = F_tau_mc(mu)
print(f"  F_tau (exact timing, MC) = {Ft:.4f}\n")
print(f"{'m':>3} {'F_m':>10} {'F_m/F_tau':>11}   optimal design (tipped fractions)")
for m in [1, 2, 3, 4, 5]:
    Fm, sig = best_design(m, mu)
    tf = 1.0 - S_of(sig, mu)
    print(f"{m:3d} {Fm:10.4f} {Fm / Ft:11.4f}   " +
          " ".join(f"{t:.3f}" for t in tf))

print("\n" + "=" * 78)
print("(5) attainability: MLE from N binary observations at sigma*  (SIG=2.222e-3)")
print("=" * 78)
mu = mu_of(2.222e-3)
z = np.linspace(-6, 6, 200001)
sg = sig_of(np.clip(mu - z, 1e-12, None))
sstar = float(sg[int(np.argmax(F_bin_analytic(sg, mu)))])
Fb = F_bin_analytic(sstar, mu)
print(f"  sigma* = {sstar:.6f},  F_bin = {Fb:.6f},  CRB at N: 1/(N F_bin)\n")
print(f"{'N':>7} {'Var(theta_hat)':>15} {'CRB':>12} {'ratio':>8}")
for N in [200, 1000, 5000, 20000]:
    R = 4000
    # true theta = 0.  Each realization tips at sigma_tip = sig_of(mu - ln u).
    U = rng.exponential(size=(R, N))
    stip = sig_of(np.clip(mu - np.log(U), 1e-12, None))
    k = np.sum(stip > sstar, axis=1)          # tipped already at sigma* (sigma decreasing)
    # MLE: solve p(sigma* - theta) = k/N  for theta
    kh = np.clip(k / N, 1e-9, 1 - 1e-9)
    g = lambda th: 1.0 - S_of(np.clip(sstar - th, 1e-9, None), mu)
    th = np.array([brentq(lambda t, ph=ph: g(t) - ph, -sstar + 1e-6, 3.0)
                   for ph in kh])
    var = float(np.var(th))
    crb = 1.0 / (N * Fb)
    print(f"{N:7d} {var:15.6e} {crb:12.6e} {var / crb:8.4f}")
