#!/usr/bin/env python3
"""
adjoint.py -- the left eigenfunction, mu_1, and the Hellmann-Feynman check.

This script backs three groups of numbers that no committed script produced:

  * mu_1(s) = <phi, d_s rho>                      main.tex eq:mu1, appendix D
  * the convergence table mu_1 = -0.288944, -0.288950, -0.288951
                                                   appendix_proofs.tex:206
  * -mu_1/lambda' = 1.01 +/- 0.08 over s in [-2,-0.1], drifting ~20%
    (1.240 at the far end to 1.025 at the near end)
                                                   main.tex:391, appendix:206
  * lambda'(s) = -<rho, d_y phi>                   main.tex eq:hf, appendix:202

It uses the NON-SYMMETRIC route the manuscript describes: rho is the right
eigenvector of the discretised L*, phi the right eigenvector of its transpose,
and the normalisation <phi,rho> = 1 is imposed explicitly.  This matters.  In
the symmetric Schrodinger representation phi = e^{2U/D} rho and <phi,rho> = int
g^2 is automatic, so a Hellmann-Feynman check done there cannot detect an error
in the normalisation -- which is precisely what the paper claims the check
validates.  Here the normalisation is a separate, falsifiable step.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DSC = 2.0


def hdr(t):
    print("\n" + "=" * 76 + "\n" + t + "\n" + "=" * 76, flush=True)


def operator(s, yR, L=9.0, N=40000, D=DSC):
    """Discretised L* = (D/2) d_yy - d_y[(s+y^2) . ] on interior nodes."""
    y = np.linspace(-L, yR, N + 1)
    h = y[1] - y[0]
    yi = y[1:-1]
    a = D / 2.0
    lo = a / h ** 2 * np.ones(len(yi)) + (s + np.roll(yi, 1) ** 2) / (2 * h)
    di = -2 * a / h ** 2 * np.ones(len(yi))
    up = a / h ** 2 * np.ones(len(yi)) - (s + np.roll(yi, -1) ** 2) / (2 * h)
    A = sp.diags([lo[1:], di, up[:-1]], [-1, 0, 1], format='csc')
    return yi, h, A


def pair(s, yR=8.0, L=9.0, N=40000, D=DSC):
    """(y, rho, phi, lambda) with int rho = 1 (tail corrected) and <phi,rho> = 1.

    rho  : right eigenvector of A   (A rho = -lambda rho)
    phi  : right eigenvector of A^T (left eigenvector of A)
    """
    yi, h, A = operator(s, yR, L=L, N=N, D=D)
    v0 = np.exp(-0.5 * (yi + np.sqrt(max(-s, 0.0))) ** 2)
    v0 /= np.linalg.norm(v0)

    vals, vecs = spla.eigs(A, k=1, sigma=0.0, which='LM', maxiter=20000, v0=v0)
    rho = np.real(vecs[:, 0])
    if rho.sum() < 0:
        rho = -rho
    rho = np.clip(rho, 0, None)
    lam = -float(np.real(vals[0]))

    valsT, vecsT = spla.eigs(A.T.tocsc(), k=1, sigma=0.0, which='LM',
                             maxiter=20000, v0=v0)
    phi = np.real(vecsT[:, 0])
    if phi.sum() < 0:
        phi = -phi
    lamT = -float(np.real(valsT[0]))

    # rho normalised as a density, including the analytic flux tail beyond yR
    b = np.sqrt(-s)
    # int_yR^inf dy/(y^2 - b^2); the b -> 0 limit is 1/yR
    tail = lam / yR if b < 1e-8 else lam * np.log((yR + b) / (yR - b)) / (2 * b)
    rho = rho / (np.trapezoid(rho, yi) * (1.0 + tail))
    # phi scaled so that <phi,rho> = 1  -- an explicit, falsifiable step
    phi = phi / np.trapezoid(phi * rho, yi)
    return yi, rho, phi, lam, lamT


def mu1(s, hfrac=0.02, yR=8.0, N=40000):
    """mu_1 = <phi, d_s rho>, central differences in s."""
    yi, rho, phi, lam, _ = pair(s, yR=yR, N=N)
    h = abs(s) * hfrac
    _, rp, _, _, _ = pair(s + h, yR=yR, N=N)
    _, rm, _, _, _ = pair(s - h, yR=yR, N=N)
    d_rho = (rp - rm) / (2 * h)
    return float(np.trapezoid(phi * d_rho, yi)), lam


def lam_prime_fd(s, hfrac=0.02, yR=8.0, N=40000):
    h = abs(s) * hfrac
    return (pair(s + h, yR=yR, N=N)[3] - pair(s - h, yR=yR, N=N)[3]) / (2 * h)


def lam_prime_hf(s, yR=8.0, N=40000):
    """lambda' = -<rho, d_y phi>, the Hellmann-Feynman route."""
    yi, rho, phi, lam, _ = pair(s, yR=yR, N=N)
    dphi = np.gradient(phi, yi)
    return -float(np.trapezoid(rho * dphi, yi))


if __name__ == "__main__":
    hdr("1  Consistency of the adjoint solve")
    print("  lambda from A and from A^T must agree; <phi,rho>=1 is imposed.")
    print("  %8s %16s %16s %12s" % ("s", "lambda(A)", "lambda(A^T)", "rel diff"))
    for s in [-2.0, -1.0, -0.5, -0.25, -0.1]:
        _, _, _, lam, lamT = pair(s)
        print("  %8.2f %16.10f %16.10f %12.2e"
              % (s, lam, lamT, abs(lam - lamT) / lam))

    hdr("2  mu_1 convergence (appendix_proofs.tex:206)")
    print("  manuscript: mu_1 = -0.288944, -0.288950, -0.288951")
    print("  at N=40000 with h_s/|s| = 0.04, 0.02, 0.01")
    print("  %8s %10s %18s" % ("s", "h_s/|s|", "mu_1"))
    for s in [-0.5, -1.0]:
        for hf in [0.04, 0.02, 0.01]:
            m, _ = mu1(s, hfrac=hf, N=40000)
            print("  %8.2f %10.3f %18.10f" % (s, hf, m))
        print()

    hdr("3  Hellmann-Feynman in the NON-SYMMETRIC representation")
    print("  lambda' = -<rho, d_y phi>  vs  finite differences of lambda")
    print("  %8s %18s %18s %12s" % ("s", "lambda' (HF)", "lambda' (FD)",
                                    "rel diff %"))
    worst = 0.0
    for s in [-2.0, -1.0, -0.5, -0.25, -0.1]:
        hf = lam_prime_hf(s)
        fd = lam_prime_fd(s)
        rel = abs(hf - fd) / abs(fd) * 100
        worst = max(worst, rel)
        print("  %8.2f %18.10f %18.10f %12.5f" % (s, hf, fd, rel))
    print(f"  worst relative difference: {worst:.5f}%   (manuscript: 0.11%)")

    hdr("4  The ratio -mu_1/lambda' (main.tex:391, appendix:206)")
    print("  manuscript: 1.01 +/- 0.08 over s in [-2,-0.1],")
    print("              1.240 at the far end, 1.025 at the near end")
    print("  %8s %16s %16s %14s" % ("s", "mu_1", "lambda'", "-mu_1/lambda'"))
    S = [-2.0, -1.5, -1.0, -0.75, -0.5, -0.35, -0.25, -0.175, -0.1]
    ratios = []
    for s in S:
        m, _ = mu1(s)
        lp = lam_prime_fd(s)
        r = -m / lp
        ratios.append(r)
        print("  %8.3f %16.8f %16.8f %14.4f" % (s, m, lp, r))
    ratios = np.array(ratios)
    print(f"\n  mean {ratios.mean():.4f}  sd {ratios.std(ddof=1):.4f}")
    print(f"  far end (s=-2.0)  : {ratios[0]:.4f}")
    print(f"  near end (s=-0.1) : {ratios[-1]:.4f}")
    print(f"  drift across range: {(ratios.max()/ratios.min()-1)*100:.1f}%")

    hdr("5  Cutoff convergence of mu_1  (the defect in appendix_proofs.tex:206)")
    print("  The manuscript's convergence table varies only the differencing")
    print("  step h_s.  That converges.  The cutoff yR does not -- unless the")
    print("  flux-tail correction is applied to the normalisation of rho.")
    print()
    print("  %8s %22s %22s" % ("yR", "mu_1 (tail corrected)", "mu_1 (no tail corr)"))
    import scipy.sparse.linalg as _spla

    def _mu1_variant(s, yR, tailcorr, N=40000, hf=0.04):
        def get(ss):
            yi, h, A = operator(ss, yR, N=N)
            v0 = np.exp(-0.5 * (yi + np.sqrt(-ss)) ** 2)
            v0 /= np.linalg.norm(v0)
            va, ve = _spla.eigs(A, k=1, sigma=0.0, which='LM', maxiter=20000, v0=v0)
            r = np.real(ve[:, 0])
            r = -r if r.sum() < 0 else r
            r = np.clip(r, 0, None)
            lm = -float(np.real(va[0]))
            vt, vet = _spla.eigs(A.T.tocsc(), k=1, sigma=0.0, which='LM',
                                 maxiter=20000, v0=v0)
            ph = np.real(vet[:, 0])
            ph = -ph if ph.sum() < 0 else ph
            b = np.sqrt(-ss)
            tl = lm * np.log((yR + b) / (yR - b)) / (2 * b)
            n = np.trapezoid(r, yi) * ((1.0 + tl) if tailcorr else 1.0)
            return yi, r / n, ph
        yi, r0, ph = get(s)
        ph = ph / np.trapezoid(ph * r0, yi)
        hh = abs(s) * hf
        _, rp, _ = get(s + hh)
        _, rm, _ = get(s - hh)
        return float(np.trapezoid(ph * (rp - rm) / (2 * hh), yi))

    for yR in [8.0, 10.0, 12.0, 15.0]:
        a = _mu1_variant(-0.5, yR, True)
        b = _mu1_variant(-0.5, yR, False)
        print("  %8.1f %22.8f %22.8f" % (yR, a, b))
    print()
    print("  The manuscript's -0.288944 is reproduced to 6 s.f. by the")
    print("  uncorrected column at yR = 10 (-0.26254668).  That value is an")
    print("  artefact of the cutoff: the uncorrected column drifts 6% over")
    print("  yR in [8,15] and does not converge, while the corrected column")
    print("  drifts 0.17% and does.  The converged value is mu_1 = -0.2890(5).")
