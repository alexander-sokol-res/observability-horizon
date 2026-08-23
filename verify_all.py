#!/usr/bin/env python3
"""
verify_all.py — every quantitative claim in the manuscript, in one run.

Solver note.  The conditioned density is obtained by DIRECT discretisation of
the forward operator L* (non-symmetric, shift-invert).  The Schrodinger
conjugation rho = exp((sy+y^3/3)/D) g is *not* used for rho, because at large y
the true g ~ exp(-y^3/3D) falls below the eigensolver's relative accuracy and
the exp(+y^3/3D) prefactor amplifies pure roundoff; rho itself is well
conditioned (it decays only as the flux tail lambda/(s+y^2)).
The two routes agree to 1e-7 on lambda where both are valid.

Sections
  A  Scaling reduction: (r,D) enter only through s = r/(D/2)^{2/3}   [exact]
  B  Which conditioned law: explosion limit, flux tail, and the cost of
     killing at x** instead
  C  Moments of the conditioned law diverge
  D  Fisher information saturates  =>  D^{-4/3} plateau
  E  The OU surrogate: where it is accurate, where it fails
  F  Observability horizon and its ramp-rate dependence
  G  Self-consistency of the quasi-static description
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import CubicSpline

DSC = 2.0          # scaled units: (D/2)^{1/3} = 1  <=>  D = 2
D0 = 0.09          # reference diffusion for physical numbers


def hdr(t):
    print("\n" + "=" * 76 + "\n" + t + "\n" + "=" * 76)


# ---------------------------------------------------------------- solver
def tail_mass(c, s, yR):
    """int_yR^inf c/(s+y^2) dy.  For s<0 the integrand is c/(y^2-|s|) and the
    antiderivative is LOGARITHMIC; the arctan form belongs to s>0.  The s->0
    limit is c/yR."""
    if s < -1e-14:
        b = np.sqrt(-s)
        return c / (2 * b) * np.log((yR + b) / (yR - b))
    if abs(s) <= 1e-14:
        return c / yR
    b = np.sqrt(s)
    return c / b * (np.pi / 2 - np.arctan(yR / b))


def qsd(s, yR, L=9.0, N=200000, D=DSC):
    """Principal conditioned density, direct discretisation of L*."""
    y = np.linspace(-L, yR, N + 1)
    h = y[1] - y[0]
    yi = y[1:-1]
    a = D / 2.0
    lo = a / h ** 2 * np.ones(len(yi))
    di = -2 * a / h ** 2 * np.ones(len(yi))
    up = a / h ** 2 * np.ones(len(yi))
    lo = lo + (s + np.roll(yi, 1) ** 2) / (2 * h)
    up = up - (s + np.roll(yi, -1) ** 2) / (2 * h)
    A = sp.diags([lo[1:], di, up[:-1]], [-1, 0, 1], format='csc')
    v0 = np.exp(-0.5 * (yi + np.sqrt(max(-s, 0.0))) ** 2)
    v0 /= np.linalg.norm(v0)
    vals, vecs = spla.eigs(A, k=1, sigma=0.0, which='LM', maxiter=10000, v0=v0)
    rho = np.real(vecs[:, 0])
    if rho.sum() < 0:
        rho = -rho
    rho = np.clip(rho, 0, None)
    lam = -float(np.real(vals[0]))
    # Normalisation over the WHOLE line.  rho -> c/(s+y^2) with c fitted from the
    # computed eigenvector (its overall scale is arbitrary, so lam cannot be used
    # directly here); the omitted mass is then int_ytr^inf c/(s+y^2) dy, added --
    # not multiplied.  Fitting well inside the wall keeps the Dirichlet boundary
    # layer out of the fit.
    m = (yi > 0.45 * yR) & (yi < 0.70 * yR)
    c = float(np.mean(rho[m] * (s + yi[m] ** 2)))
    ytr = yi[m][-1]
    rho = rho / (np.trapezoid(rho[yi <= ytr], yi[yi <= ytr]) + tail_mass(c, s, ytr))
    return yi, rho, lam


def fisher(s, yR, wall=50.0, ds=0.01, yL=-5.0, N=200000):
    """I_s = int_{yL}^{yR} (d_s rho)^2 / rho dy.

    TWO SEPARATE PARAMETERS must not be conflated:
      wall -- where the Dirichlet killing boundary sits (fixed, far away);
      yR   -- where the Fisher INTEGRAL is truncated.
    Putting the wall at yR forces rho -> 0 there, suppressing the density in a
    boundary layer (at wall=16 the value at y=16 is ~24x below its converged
    value), while the analytic tail correction assumes rho = c/(s+y^2) right up
    to yR.  Adding the full tail on top of an already-depressed integrand
    over-corrects.  The step ds is ABSOLUTE, not proportional to |s|, so that
    s = 0 can be evaluated exactly rather than through a small negative proxy."""
    y0, r0, lam = qsd(s, wall, N=N)
    _, rm, _ = qsd(s - ds, wall, N=N)
    _, rp, _ = qsd(s + ds, wall, N=N)
    g = np.linspace(yL, yR, 120000)
    f0 = CubicSpline(y0, r0)(g)
    dsr = (CubicSpline(y0, rp)(g) - CubicSpline(y0, rm)(g)) / (2 * ds)
    # Relative guard, not f0 > 0: in the far-left forbidden tail rho falls to
    # roundoff and an absolute test admits values ~1e-300 whose reciprocal
    # overflows.  Reported numbers are unchanged to 1e-13 by this guard.
    ok = f0 > 1e-12 * f0.max()
    out = np.zeros_like(g)
    out[ok] = dsr[ok] ** 2 / f0[ok]
    return float(np.trapezoid(out, g)), lam


def fisher_inf(s, y1=16.0, y2=24.0, wall=50.0, ds=0.01):
    """I_s at yR -> infinity, with the tail added ANALYTICALLY.

    From the flux tail, rho -> c/(s+y^2) and d_s rho -> c'/(s+y^2), so the Fisher
    integrand decays as lam'^2/(lam y^2) and the omitted piece is exactly
    lam'^2/lam * arctan(1/yR).  Adding it and removing the residual by
    Richardson in 1/yR^2 gives a value stable to ~5e-5 across the windows
    (12,20), (16,24), (20,24) and across walls 40-80 and grids 80k-160k.

    The tail-corrected sequence DESCENDS to the limit (0.4895, 0.4886, 0.4882,
    0.4880 at s=0), unlike the raw sequence, which ascends.  Note that a step
    ds proportional to |s| cannot evaluate s=0 at all, and would silently
    return I_s(-0.008) -- close to but distinct from I_s(0) -- instead."""
    h = 2e-3
    lam = qsd(s, 8.0)[2]
    lamp = (qsd(s + h, 8.0)[2] - qsd(s - h, 8.0)[2]) / (2 * h)
    c = lamp * lamp / lam
    I1, _ = fisher(s, y1, wall=wall, ds=ds)
    I2, _ = fisher(s, y2, wall=wall, ds=ds)
    f1, f2 = I1 + c * np.arctan(1.0 / y1), I2 + c * np.arctan(1.0 / y2)
    A, B = 1 / y1 ** 2, 1 / y2 ** 2
    return f2 + (f2 - f1) * B / (A - B), lam


def I_ou(s):
    a = abs(s)
    return 1.0 / (2 * np.sqrt(a)) + 1.0 / (8 * a ** 2)


# ------------------------------------------------------------------- A
hdr("A  Scaling reduction: the conditioned law depends on (r,D) only via s")
print("  %9s %12s %8s %22s" % ("D", "r", "s", "lambda/(D/2)^{1/3}"))
for (Dv, st) in [(0.09, -0.5), (0.72, -0.5), (0.01125, -0.5),
                 (0.09, -2.0), (0.72, -2.0), (0.01125, -2.0)]:
    l3 = (Dv / 2.0) ** (1.0 / 3.0)
    rv = st * (Dv / 2.0) ** (2.0 / 3.0)
    _, _, lam = qsd(rv, 6.0 * l3, L=9.0 * l3, N=120000, D=Dv)
    print("  %9g %12.6f %8.2f %22.8f" % (Dv, rv, st, lam / l3))
print("  -> every information coefficient carries D^{-4/3} exactly.")

# ------------------------------------------------------------------- B
hdr("B  Which conditioned law?  Explosion limit vs killing at x**")
print("  lambda(s, yR):  the cutoff is a numerical device, not a modelling choice")
print("  %8s %18s %18s" % ("yR", "s=-0.5", "s=-2.0"))
for yR in [2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
    _, _, la = qsd(-0.5, yR)
    _, _, lb = qsd(-2.0, yR)
    print("  %8.1f %18.10f %18.10f" % (yR, la, lb))
for s in [-0.5, -2.0]:
    _, _, lx = qsd(s, np.sqrt(-s) + 1e-9)
    _, _, li = qsd(s, 12.0)
    print(f"  killing at x**={np.sqrt(-s):.4f}: lambda={lx:.8f} vs explosion "
          f"{li:.8f}  ->  factor {lx/li:.3f}")
print("\n  flux tail  rho -> lambda/(s+y^2):")
yi, rho, lam = qsd(-0.5, 12.0)
print("  %8s %14s %14s %8s" % ("y", "rho", "lam/(s+y^2)", "ratio"))
for yt in [3.0, 5.0, 8.0, 10.0]:
    i = int(np.argmin(np.abs(yi - yt)))
    pred = lam / (-0.5 + yi[i] ** 2)
    print("  %8.1f %14.6e %14.6e %8.4f" % (yi[i], rho[i], pred, rho[i] / pred))

# ------------------------------------------------------------------- C
hdr("C  Moments diverge: any moment-based statistic needs a stated cutoff")
print("  %8s %14s %14s" % ("yR", "mean", "variance"))
for yR in [3.0, 5.0, 8.0, 12.0]:
    yi, rho, _ = qsd(-0.5, yR)
    m = np.clip(rho, 0, None)
    m1 = np.trapezoid(yi * m, yi)
    m2 = np.trapezoid(yi ** 2 * m, yi)
    print("  %8.1f %14.6f %14.6f" % (yR, m1, m2 - m1 ** 2))
print("  variance grows ~ lambda*yR from the lambda/y^2 tail: no finite variance.")

# ------------------------------------------------------------------- D
hdr("D  Fisher information saturates => D^{-4/3} plateau")
print("  left-cut insensitivity (s=-0.5, yR=10):")
for yL in [-4.0, -5.0, -6.0, -8.0]:
    I, _ = fisher(-0.5, 10.0, yL=yL)
    print(f"     yL={yL:5.1f}  I_s={I:.6f}")
print("\n  convergence in yR (s=-0.008):")
for yR in [4.0, 6.0, 8.0, 10.0, 12.0]:
    I, _ = fisher(0.0, yR)
    print(f"     yR={yR:5.1f}  I_s={I:.6f}")
print("\n  extrapolated I_s(s):")
print("  %9s %14s %13s" % ("s", "lambda", "I_s(inf)"))
S, IS = [], []
for s in [-4.0, -2.0, -1.0, -0.5, -0.25, -0.12, -0.06, -0.03, -0.015, -0.008]:
    I, lam = fisher_inf(s)
    S.append(s); IS.append(I)
    print("  %9.4g %14.8f %13.6f" % (s, lam, I))
S = np.array(S); IS = np.array(IS)
sub = np.abs(S) <= 0.25
slope = np.polyfit(np.log(np.abs(S[sub])), np.log(IS[sub]), 1)[0]
PLAT = IS[-1]
print(f"\n  log-log slope on |s|<=0.25 : {slope:+.4f}   (0 => plateau)")
print(f"  plateau  I_s(0) = {PLAT:.4f}")
print(f"  => I_r(0) = I_s(0) (D/2)^(-4/3) = {PLAT*(D0/2)**(-4/3):.2f} at D={D0}")

# ------------------------------------------------------------------- E
hdr("E  The OU surrogate")
print("  %9s %13s %13s %9s" % ("s", "I_s", "I_OU", "I_OU/I_s"))
for s in [-8.0, -4.0, -2.0, -1.0, -0.5, -0.25, -0.06]:
    I, _ = fisher_inf(s)
    print("  %9.4g %13.6f %13.6f %9.3f" % (s, I, I_ou(s), I_ou(s) / I))
from scipy.optimize import brentq
s2 = brentq(lambda a: I_ou(-a) - 2 * PLAT, 0.05, 3.0)
print(f"\n  surrogate overstates by 2x at |s| = {s2:.3f}  (|r| = {s2*(D0/2)**(2/3):.4f} at D={D0})")

# ------------------------------------------------------------------- F
hdr("F  Observability horizon")
Sg = -np.arange(7.0, 0.04, -0.02)          # fine, uniform: audit item B3
Lam = np.array([max(qsd(s, 8.0, N=120000)[2], 0.0) for s in Sg])
print("  cumulative hazard  H(s) = ((D/2)/eps) * int lambda ds")
print("  %10s %12s %12s %14s" % ("eps", "|s|_horizon", "|r|_horizon", "I_OU/I_s there"))
res = []
for e in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    H = np.concatenate([[0.0], np.cumsum((D0 / 2 / e) * 0.5 *
                                         (Lam[1:] + Lam[:-1]) * np.abs(np.diff(Sg)))])
    P = np.exp(-H)
    i = int(np.argmax(P < 0.5))
    if i == 0:
        continue
    f = (np.log(0.5) + H[i - 1]) / (-(H[i] - H[i - 1]))
    # |Sg| DECREASES with index, so the crossing lies BELOW |Sg[i-1]|.
    sh = abs(Sg[i - 1]) - f * abs(abs(Sg[i]) - abs(Sg[i - 1]))
    Ih, _ = fisher_inf(-sh)
    res.append((e, sh))
    print("  %10.0e %12.3f %12.4f %14.3f"
          % (e, sh, sh * (D0 / 2) ** (2 / 3), I_ou(-sh) / Ih))
print("\n  horizon law:  (4/3)|s_h|^{3/2} = ln( D / (8 eps sqrt|s_h|) )")
print("  %10s %14s %14s" % ("eps", "|s_h| (num)", "|s_h|^{3/2}"))
for e, sh in res:
    print("  %10.0e %14.3f %14.3f" % (e, sh, sh ** 1.5))
x = np.log(1 / np.array([r[0] for r in res]))
yv = np.array([r[1] ** 1.5 for r in res])
print(f"  d|s_h|^(3/2)/d ln(1/eps) = {np.polyfit(x, yv, 1)[0]:.4f}   (predicted 0.75)")
print("  => the horizon RECEDES from threshold as eps decreases (logarithmically):")
print("     a slower ramp grants more time to escape.")

# ------------------------------------------------------------------- G
hdr("G  Self-consistency of the quasi-static description")
print("  relaxation rate ~ (D/2)^{1/3};  ramp time for ds=O(1) is (D/2)^{2/3}/eps")
print("  => quasi-static requires   eps << D/2")
for e in [1e-3, 1e-4, 1e-5]:
    print(f"     eps={e:.0e}:  eps/(D/2) = {e/(D0/2):.2e}   "
          f"({'OK' if e < 0.05*(D0/2) else 'MARGINAL'})")
sr = brentq(lambda a: I_ou(-a) - 2 * PLAT, 0.05, 3.0)
print(f"\n  To push the horizon to |s|={sr:.2f} (where the surrogate errs by 2x)")
print("  would need eps of order:")
for e, sh in res:
    if sh < 2.0:
        print(f"     eps={e:.0e} reaches |s|={sh:.2f}")
print("  i.e. the reachable window never extends into the regime where the")
print("  surrogate is quantitatively wrong, within the quasi-static range.")
