"""Asymptotic scaling of the fold-location Fisher information, F_fold ~ 4|s_h|.

Computes F_fold(eps_hat) for the quasi-stationary escape rate (exact eigenvalue,
not the Kramers approximation) and compares it against the leading-order
prediction 4|s_h(eps_hat)| from the ramp-rate horizon.  lambda(s) spans many
decades in magnitude and its eigensolver estimate loses precision once it drops
below the solver's cancellation floor, so the usable domain is truncated at
lambda > 1e-8 and the spline is fit to log(lambda) rather than lambda itself.
The normalization Z (should equal 1) is reported as a check that the truncated
domain still captures essentially all of the escape-time distribution.
"""
import numpy as np
import scipy.linalg as sla
from scipy.interpolate import CubicSpline


def lam(s, L=12.0, N=16000):
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


S = np.linspace(-5.0, 2.5, 301)
L = np.array([lam_richardson(x) for x in S])
ok = L > 1e-8
S, L = S[ok], L[ok]
print(f"usable range s in [{S.min():.2f},{S.max():.2f}], "
      f"lambda in [{L.min():.3e},{L.max():.3f}]")
spline = CubicSpline(S, np.log(L))


def F_fold(eps_hat, n=400000):
    s = np.linspace(S.min(), S.max(), n)
    ds = s[1] - s[0]
    lm = np.exp(spline(s))
    H = np.cumsum(lm) * ds / eps_hat
    P = np.exp(-H)
    f = lm * P / eps_hat
    score = spline(s, 1) - lm / eps_hat
    Z = float(np.sum(f) * ds)
    return float(np.sum(score ** 2 * f) * ds), Z


def horizon(eps_hat):
    return (0.75 * np.log(1 / (2 * np.pi * np.log(2) * eps_hat))) ** (2 / 3)


print(f"\n{'eps_hat':>10} {'|s_h|':>8} {'F_fold':>10} {'4|s_h|':>9} "
      f"{'ratio':>7} {'norm Z':>8}")
for eh in [0.05, 0.0222, 5e-3, 2.22e-3, 1e-3]:
    F, Z = F_fold(eh)
    s = horizon(eh)
    flag = '' if 0.98 < Z < 1.02 else '   <-- domain too narrow, ignore'
    print(f"{eh:10.2e} {s:8.4f} {F:10.4f} {4 * s:9.4f} {F / (4 * s):7.4f} "
          f"{Z:8.4f}{flag}")
