#!/usr/bin/env python3
"""
observability_horizon.py -- reference implementation of the observability
horizon of the accompanying paper.

Given a noise level and a ramp rate (or a multidimensional model and a noise
covariance), this returns the closest approach to a saddle-node at which half a
ramped ensemble is still present, together with every validity check the paper
attaches to that number.  It is meant to be usable without reading the paper,
and to refuse to answer where the paper's own analysis does not support one.

Usage
-----
    python3 observability_horizon.py --D 0.09 --eps 1e-3
    python3 observability_horizon.py --D 0.09 --eps 1e-3 --lead-time 300
    python3 observability_horizon.py --selftest

The multidimensional entry point is `horizon_nd`; see the Stommel self-test.
"""
import argparse
import numpy as np

LN2 = np.log(2.0)
SIG_DOMAIN = 1.0 / (2 * np.pi * LN2)   # 0.2296  -- Eq. (15): closed form defined
SIG_CONT = 0.10                        # Eq. (24): median stays outside |s|_2x
SIG_KZ = 0.201                         # Sec. VI B: quasi-static reading safe
TAU_EFF = 1.14                         # Sec. VI E: adiabatic lag coefficient
S2X = 0.611                            # Eq. (21): OU wrong by 2x inside this
PLATEAU = 0.48752                      # Eq. (20): I_s(0)


def horizon_scaled(SIG):
    """|s_h| from Eq. (13).  Quasi-static; see `horizon` for the lag."""
    if SIG >= SIG_DOMAIN:
        return np.nan
    return (0.75 * np.log(1.0 / (2 * np.pi * LN2 * SIG))) ** (2.0 / 3.0)


def corr_budget(SIG):
    """Eq. (18): relaxation times remaining at the horizon.  D-independent."""
    if SIG >= SIG_DOMAIN:
        return np.nan
    return 1.5 / SIG * np.log(1.0 / (2 * np.pi * LN2 * SIG))


def horizon(D, eps, lag_correct=True):
    """Full report for the 1-D fold  dx = (r+x^2)dt + sqrt(D) dW,  dr/dt = eps."""
    SIG = 2.0 * eps / D
    ell = (D / 2.0) ** (1.0 / 3.0)
    sh_qs = horizon_scaled(SIG)
    # The lag is O(SIG) and is subtracted in SCALED units (Sec. VI E).
    sh = sh_qs - TAU_EFF * SIG if (lag_correct and np.isfinite(sh_qs)) else sh_qs
    out = {
        "SIG": SIG, "ell": ell,
        "s_h_quasistatic": sh_qs,
        "s_h": sh,
        "r_h": ell ** 2 * sh if np.isfinite(sh) else np.nan,
        "lag_scaled": TAU_EFF * SIG,
        "lead_time": (ell ** 2 * sh) / eps if np.isfinite(sh) else np.nan,
        "corr_budget": corr_budget(SIG),
        "in_domain": SIG < SIG_DOMAIN,
        "quasistatic_ok": SIG < SIG_KZ,
        "median_contained": bool(np.isfinite(sh) and sh > S2X),
        "surrogate_ok_here": bool(np.isfinite(sh) and sh > S2X),
    }
    return out


def reduce_nd(F, J, e, f, Sigma, b_coeff):
    """Noise projection of Eq. (40): D_eff = b^2 f^T Sigma Sigma^T f.

    e, f are the right and LEFT null vectors of the Jacobian at the fold, with
    f^T e = 1.  Using the right vector here is the common mistake; the left
    vector is the one that filters the noise.
    """
    f = np.asarray(f, float)
    P2 = float(f @ (Sigma @ Sigma.T) @ f)
    return b_coeff ** 2 * P2, P2


def report(D, eps, lead_time=None):
    h = horizon(D, eps)
    print(f"  D = {D:g},  eps = {eps:g}   ->   SIG = 2 eps/D = {h['SIG']:.5g}")
    if not h["in_domain"]:
        print(f"  REFUSED: SIG >= {SIG_DOMAIN:.4f}; the closed form is not defined")
        print("  (the ramp outruns escape -- no horizon exists ahead of the fold)")
        return h
    print(f"  |s_h| quasi-static      = {h['s_h_quasistatic']:.4f}")
    print(f"  adiabatic lag           = {h['lag_scaled']:.4f}"
          f"   ({100*h['lag_scaled']/h['s_h_quasistatic']:.1f}% of the horizon)")
    print(f"  |s_h| lag-corrected     = {h['s_h']:.4f}")
    print(f"  |r_h| (physical units)  = {h['r_h']:.6g}")
    print(f"  lead time to threshold  = {h['lead_time']:.6g}")
    print(f"  relaxation times left   = {h['corr_budget']:.4g}   [Eq. (18)]")
    print(f"  quasi-static reading OK = {h['quasistatic_ok']}  (needs SIG < {SIG_KZ})")
    print(f"  OU surrogate usable     = {h['median_contained']}"
          f"  (needs |s_h| > {S2X}, i.e. SIG < ~{SIG_CONT})")
    if not h["median_contained"]:
        print("  WARNING: the median trajectory reaches inside |s|_2x, so an OU-based")
        print("           indicator is being applied where it errs by >2x.")
    if lead_time is not None:
        ok = lead_time <= h["lead_time"]
        print(f"\n  claimed lead time {lead_time:g}: "
              f"{'ACHIEVABLE' if ok else 'BEYOND THE HORIZON'}"
              f"  (max {h['lead_time']:.6g})")
    return h


def selftest():
    ok = True
    print("1. Table III of the paper (quasi-static |s_h|, D = 0.09)")
    for eps, want in [(1e-3, 1.4530), (1e-4, 2.2960), (1e-5, 3.0040),
                      (1e-6, 3.6360), (1e-7, 4.2170)]:
        got = horizon_scaled(2 * eps / 0.09)
        good = abs(got - want) < 5e-4
        ok &= good
        print(f"   eps={eps:.0e}: {got:.4f} vs {want:.4f}  {'ok' if good else 'FAIL'}")

    print("2. Eq. (18) is D-independent at fixed SIG")
    vals = [corr_budget(2 * e / D) for D, e in
            [(0.09, 1e-3), (2.0, 2.22e-2), (0.01, 1.11e-4)]]
    good = max(vals) - min(vals) < 0.5
    ok &= good
    print(f"   spread over a 200x range in D: {max(vals)-min(vals):.3g}  "
          f"{'ok' if good else 'FAIL'}")

    print("3. Domain refusal above Eq. (15)")
    good = not np.isfinite(horizon_scaled(0.25))
    ok &= good
    print(f"   SIG=0.25 -> nan  {'ok' if good else 'FAIL'}")

    print("4. Containment flag flips near SIG = 0.10 (Eq. 24)")
    lo = horizon(0.09, 0.09 * 0.08 / 2)["median_contained"]
    hi = horizon(0.09, 0.09 * 0.122 / 2)["median_contained"]
    good = lo and not hi
    ok &= good
    print(f"   SIG=0.08 contained={lo}, SIG=0.122 contained={hi}  "
          f"{'ok' if good else 'FAIL'}")

    print("5. Stommel noise projection (Sec. VII, Appendix E)")
    # Actual left null vector at the fold, from stommel_horizon.py step 2.
    f = np.array([+0.83338351, -1.67647713])
    b_coeff = 0.3353545916
    # Salinity-only forcing, as in Sec. VII: Sigma = diag(0, sigma), sigma = 1.
    Sigma = np.diag([0.0, 1.0])
    Deff, P2 = reduce_nd(None, None, None, f, Sigma, b_coeff)
    want_P2 = f[1] ** 2                     # = sigma^2 f_2^2, script line 23
    good = abs(P2 - want_P2) < 1e-9
    ok_local = good
    print(f"   f^T Sigma Sigma^T f = {P2:.8f}  vs  f_2^2 = {want_P2:.8f}  "
          f"{'ok' if good else 'FAIL'}")
    # Using the RIGHT null vector instead is the mistake the paper warns about:
    e = np.array([-0.51926107, -0.85461567])
    print(f"   (right null vector would give {float(e @ (Sigma@Sigma.T) @ e):.6f} "
          f"-- wrong by {abs(P2/float(e @ (Sigma@Sigma.T) @ e)):.2f}x)")
    ok &= ok_local

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--D", type=float, help="diffusion coefficient")
    ap.add_argument("--eps", type=float, help="ramp rate dr/dt")
    ap.add_argument("--lead-time", type=float, default=None,
                    help="a claimed early-warning lead time, to test against the horizon")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if a.D is None or a.eps is None:
        ap.error("need --D and --eps (or --selftest)")
    report(a.D, a.eps, a.lead_time)
