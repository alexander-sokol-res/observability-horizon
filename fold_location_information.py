"""Fisher information for the fold location, computed with the exact escape
rate rather than the Kramers approximation.

The Kramers rate vanishes at the fold (its sqrt(barrier) prefactor goes to
zero as s -> 0), which makes the naive fold-location Fisher information
formally divergent as the cutoff at the fold is removed. The exact eigenvalue
does not vanish there, so the exact-rate calculation is finite and stable
under changing where the domain is truncated past the fold.
"""
import numpy as np
import scipy.linalg as sla
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


def lam(s, L=12.0, N=16000):
    """Ground-state eigenvalue of -psi'' + [(s+y^2)^2/4 + y] psi = lambda psi."""
    y = np.linspace(-L, L, N + 1)[1:-1]
    h = y[1] - y[0]
    d = 2 / h ** 2 + ((s + y ** 2) ** 2 / 4 + y)
    e = -1 / h ** 2 * np.ones(len(y) - 1)
    return sla.eigh_tridiagonal(d, e, select='i', select_range=(0, 0),
                                 eigvals_only=True)[0]


def lam_richardson(s):
    a = lam(s, N=8000)
    b = lam(s, N=16000)
    return (4 * b - a) / 3


S = np.linspace(-6.0, 3.0, 226)
LAM = np.array([lam_richardson(s) for s in S])
spline = CubicSpline(S, LAM)
print("Exact rate continues smoothly through and past the fold:")
for s in [-2, -1, -0.5, 0, 0.5, 1.0, 2.0]:
    print(f"   lambda({s:+.1f}) = {spline(s):.5f}")


def fold_info(SIG, smax, kramers=False, scut=None, n=400000):
    s = np.linspace(-6.0, smax, n)
    ds = s[1] - s[0]
    if kramers:
        lm = np.where(s < 0, np.sqrt(np.abs(s)) / np.pi
                      * np.exp(-4 / 3 * np.abs(s) ** 1.5), 0.0)
        if scut is not None:
            lm = np.where(s > -scut, 0.0, lm)
    else:
        lm = np.maximum(spline(s), 1e-300)
    H = np.cumsum(lm) * ds / SIG
    P = np.exp(-H)
    f = lm * P / SIG
    dlnl = np.gradient(np.log(np.maximum(lm, 1e-300)), ds)
    score = dlnl - lm / SIG
    return float(np.sum(score ** 2 * f) * ds), s, P, lm


print("\nFold-location information, exact rate vs Kramers:")
for SIG in [0.0222]:
    print(f"  SIG={SIG}: exact rate, continuation to s = +0.5/+1/+2:")
    for smax in [0.5, 1.0, 2.0]:
        F, _, _, _ = fold_info(SIG, smax)
        print(f"     smax={smax:+.1f}: F_fold = {F:.4f}")
    print("   Kramers rate, as the cutoff at the fold is lowered:")
    for cut in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7]:
        F, _, _, _ = fold_info(SIG, 0.0, kramers=True, scut=cut)
        print(f"     cutoff {cut:.0e}: F_fold = {F:.3f}")

print("\nOptimal binary census for the fold location (exact rate):")
for SIG in [0.0222, 0.05, 0.10]:
    F, s, P, lm = fold_info(SIG, 2.0)
    Fb = lm ** 2 * P / (SIG ** 2 * np.maximum(1 - P, 1e-300))
    i = int(np.nanargmax(Fb))
    print(f"  SIG={SIG:<7}: optimum at s={s[i]:+.3f}, "
          f"tipped fraction={1 - P[i]:.3f}, "
          f"F_bin={Fb[i]:.4f}, F_bin/F_fold={Fb[i] / F:.4f}")
print("  (compare with the universal binary-census rule, 0.7968)")
