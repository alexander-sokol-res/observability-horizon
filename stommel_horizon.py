#!/usr/bin/env python3
"""
stommel_horizon.py — the observability horizon predicted with NO fitted
parameters for a two-dimensional physical model with a genuine saddle-node.

MODEL.  Stommel two-box thermohaline circulation on the smooth branch q=T-S>0,

    dT/dt = eta1 - T (1 + q)
    dS/dt = eta2 - S (eta3 + q)  + sigma dW        q = T - S

with eta1=3, eta3=0.3 and the freshwater forcing eta2 ramped toward the fold.
Noise acts ONLY on salinity, so the effective one-dimensional diffusion is a
nontrivial projection of the applied variance, not the applied variance itself.

CHAIN.  Everything analytic; simulation is used only to check the prediction.

  1. fold (T_c,S_c,mu_c) from  d eta2/dq = 0   (machine precision)
  2. right/left null vectors  J e = 0,  f^T J = 0,  f.e = 1
  3. normal form on the centre manifold,  dz/dt = a dmu + b z^2, with
         a = f . dF/dmu = f_2                          (exact: dF/dmu=(0,1))
         b = (1/2) f . D2F[e,e] = (e2-e1)(f1 e1 + f2 e2)   (exact: F quadratic)
  4. rescale x = b z  =>  dx/dt = r + x^2 ,  r = P dmu ,  P = a b
  5. noise projection   D_z = f^T Sigma Sigma^T f = sigma^2 f_2^2
         =>  D_eff = b^2 D_z = (P sigma)^2       ramp  eps_eff = P v
  6. PREDICT   |r_h| = [ (3/8) D_eff ln( D_eff / (4 pi ln2 eps_eff) ) ]^(2/3)
  7. VERIFY    full 2-D simulation, and the reduced 1-D normal form, both with
               the same escape rule.  1-D vs 2-D isolates the reduction error;
               closed form vs 1-D isolates the Kramers error.

Note P = a b and D_eff are invariant under rescaling e -> k e (a -> a/k,
b -> k b), as they must be; this is checked explicitly (V2).
"""
import numpy as np
from scipy.optimize import brentq

ETA1, ETA3 = 3.0, 0.3
LN2 = np.log(2.0)
FOURPI_LN2 = 4 * np.pi * LN2

FAIL = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "**FAIL**"
    if not cond:
        FAIL.append(name)
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))


def hdr(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ------------------------------------------------------------------ model
def Fvec(T, S, mu):
    q = T - S
    return np.array([ETA1 - T * (1.0 + q), mu - S * (ETA3 + q)])


def Jac(T, S):
    return np.array([[-1 - 2 * T + S, T],
                     [-S, -ETA3 - T + 2 * S]])


def eta2_of_q(q):
    return (ETA1 * (ETA3 + q) - q * (1 + q) * (ETA3 + q)) / (1 + q)


# ============================================================== STEP 1
hdr("STEP 1  Locate the fold exactly")

_d = lambda q: (eta2_of_q(q + 1e-7) - eta2_of_q(q - 1e-7)) / 2e-7
QC = brentq(_d, 0.1, 0.9, xtol=1e-15, rtol=8.9e-16)
MUC = eta2_of_q(QC)
TC = ETA1 / (1 + QC)
SC = MUC / (ETA3 + QC)

res = Fvec(TC, SC, MUC)
J = Jac(TC, SC)
evals = np.linalg.eigvals(J)
i0 = int(np.argmin(np.abs(evals)))
lam_perp = float(np.real(evals[1 - i0]))

print(f"  q_c   = {QC:.12f}")
print(f"  mu_c  = {MUC:.12f}   (eta2 at the fold)")
print(f"  T_c   = {TC:.12f}")
print(f"  S_c   = {SC:.12f}")
print(f"  eigenvalues: {np.real(evals[i0]):.3e} (centre), {lam_perp:.6f} (transverse)")
check("equilibrium residual < 1e-12", np.max(np.abs(res)) < 1e-12,
      f"|F| = {np.max(np.abs(res)):.2e}")
check("zero eigenvalue < 1e-8", abs(np.real(evals[i0])) < 1e-8,
      f"lam_0 = {np.real(evals[i0]):.2e}")
check("transverse eigenvalue strictly negative", lam_perp < -0.1,
      f"lam_perp = {lam_perp:.4f}")

# ============================================================== STEP 2
hdr("STEP 2  Centre-manifold reduction (exact; F is quadratic)")


def reduction(scale=1.0):
    """Null vectors and normal-form coefficients.  `scale` rescales e to test
    invariance of the physical outputs P=ab and D_eff."""
    w, V = np.linalg.eig(J)
    k = int(np.argmin(np.abs(w)))
    e = np.real(V[:, k]) * scale
    wl, Vl = np.linalg.eig(J.T)
    kl = int(np.argmin(np.abs(wl)))
    f = np.real(Vl[:, kl])
    f = f / np.dot(f, e)                      # normalise f.e = 1
    a = f[1]                                  # dF/dmu = (0,1)
    b = (e[1] - e[0]) * (f[0] * e[0] + f[1] * e[1])
    return e, f, a, b


E, Fl, A, B = reduction()
P = A * B

# exact second derivative, for cross-check of b
D2 = np.array([-2 * E[0] ** 2 + 2 * E[0] * E[1],
               -2 * E[0] * E[1] + 2 * E[1] ** 2])
b_alt = 0.5 * float(np.dot(Fl, D2))
# finite-difference cross-check of b
h = 1e-5
d2fd = (Fvec(TC + h * E[0], SC + h * E[1], MUC)
        - 2 * Fvec(TC, SC, MUC)
        + Fvec(TC - h * E[0], SC - h * E[1], MUC)) / h ** 2
b_fd = 0.5 * float(np.dot(Fl, d2fd))

print(f"  e (right null) = [{E[0]:+.8f}, {E[1]:+.8f}]")
print(f"  f (left  null) = [{Fl[0]:+.8f}, {Fl[1]:+.8f}]     f.e = {np.dot(Fl,E):.12f}")
print(f"  a = f.dF/dmu   = {A:+.10f}")
print(f"  b (closed form)= {B:+.10f}")
print(f"  b (D2F route)  = {b_alt:+.10f}")
print(f"  b (finite diff)= {b_fd:+.10f}")
print(f"  P = a*b        = {P:+.10f}      r = P (eta2 - mu_c)")
check("b closed form == D2F route", abs(B - b_alt) < 1e-10 * max(1, abs(B)))
check("b == finite difference", abs(B - b_fd) < 1e-4 * abs(B),
      f"rel {abs(B-b_fd)/abs(B):.2e}")
check("P > 0 (two equilibria for eta2 < mu_c)", P > 0, f"P = {P:.6f}")

# V2: invariance under rescaling of e
for sc in [0.37, 2.9, -1.4]:
    _, _, a2, b2 = reduction(sc)
    check(f"P invariant under e -> {sc}e", abs(a2 * b2 - P) < 1e-10 * abs(P),
          f"P = {a2*b2:.10f}")

# ============================================================== STEP 3
hdr("STEP 3  Validate the reduction against the true equilibrium branch")
print("  normal form predicts fixed points at  x = -+ sqrt(|r|),  r = P dmu")
print("  i.e.  z = -+ sqrt(|r|)/b  with  X = X_c + z e + O(z^2)")
print(f"\n  {'dmu':>12} {'z_stable (true)':>18} {'z_stable (nf)':>16} {'ratio':>9}")
ok_branch = True
for dmu in [-1e-5, -1e-4, -1e-3, -1e-2]:
    mu = MUC + dmu
    # true equilibrium: solve eta2_of_q(q) = mu on the stable branch q > q_c
    qs = brentq(lambda q: eta2_of_q(q) - mu, QC + 1e-14, 3.0, xtol=1e-15)
    Ts = ETA1 / (1 + qs)
    Ss = mu / (ETA3 + qs)
    z_true = float(np.dot(Fl, np.array([Ts - TC, Ss - SC])))
    r = P * dmu
    z_nf = -np.sqrt(-r) / B          # stable branch
    print(f"  {dmu:12.1e} {z_true:18.8e} {z_nf:16.8e} {z_true/z_nf:9.5f}")
    if abs(dmu) <= 1e-3 and abs(z_true / z_nf - 1) > 0.05:
        ok_branch = False
check("normal form reproduces the true branch to <5% for |dmu|<=1e-3", ok_branch)

# ============================================================== STEP 4
hdr("STEP 4  Design: choose sigma and ramp rate to hit target (D_eff, SIG)")
print("  D_eff = (P sigma)^2      eps_eff = P v      SIG = 2 eps_eff / D_eff")


def design(D_eff, SIG):
    sigma = np.sqrt(D_eff) / P
    eps_eff = SIG * D_eff / 2.0
    v = eps_eff / P
    return sigma, v, eps_eff


def horizon_cf(D_eff, eps_eff):
    """Closed form, scaled and physical."""
    SIG = 2 * eps_eff / D_eff
    s_h = ((3.0 / 4.0) * np.log(1.0 / (2 * np.pi * LN2 * SIG))) ** (2.0 / 3.0)
    r_h = ((3.0 / 8.0) * D_eff * np.log(D_eff / (FOURPI_LN2 * eps_eff))) ** (2.0 / 3.0)
    return s_h, r_h


D_EFF = 1.0e-4
ELL = (D_EFF / 2) ** (1.0 / 3.0)
print(f"\n  D_eff = {D_EFF:.3e}   ell = (D/2)^(1/3) = {ELL:.6f}")
print(f"  centre-mode rate at |s|=1.5 : {2*np.sqrt(1.5*(D_EFF/2)**(2/3)):.4f}")
print(f"  transverse rate             : {abs(lam_perp):.4f}"
      f"   -> separation {abs(lam_perp)/(2*np.sqrt(1.5*(D_EFF/2)**(2/3))):.0f}x")

# ============================================================== STEP 5
hdr("STEP 5  Simulate: full 2-D Stommel, and the reduced 1-D normal form")

SEEDS = np.random.SeedSequence(20260814).spawn(8)
S_START = 3.5          # start here; hazard beyond this is negligible (checked)
Y_ESC = 8.0            # escape threshold in scaled units x/ell


def run2d(sigma, v, s_start, y_esc, dt, npath, rng, ncheck=400):
    """Full 2-D Stommel with ramped eta2 and salinity noise.
    Returns (s_grid, alive_fraction, min_q_seen)."""
    r0 = -s_start * (D_EFF / 2) ** (2.0 / 3.0)
    mu0 = MUC + r0 / P
    Tt = np.full(npath, ETA1 / (1 + QC))      # start at the stable equilibrium
    q0 = brentq(lambda q: eta2_of_q(q) - mu0, QC + 1e-14, 3.0, xtol=1e-15)
    Tt = np.full(npath, ETA1 / (1 + q0))
    St = np.full(npath, mu0 / (ETA3 + q0))
    nstep = int(abs(r0) / (P * v) / dt)
    xesc = y_esc * ELL
    alive = np.ones(npath, dtype=bool)
    every = max(1, nstep // ncheck)
    sg, af = [], []
    sq = sigma * np.sqrt(dt)
    minq = np.inf
    for k in range(nstep):
        mu = mu0 + v * (k * dt)
        q = Tt - St
        dT = (ETA1 - Tt * (1.0 + q)) * dt
        dS = (mu - St * (ETA3 + q)) * dt + sq * rng.standard_normal(npath)
        Tt = Tt + dT
        St = St + dS
        # escape in the normal-form coordinate x = b f.(X - X_c)
        x = B * (Fl[0] * (Tt - TC) + Fl[1] * (St - SC))
        alive &= (x < xesc)
        if k % every == 0:
            minq = min(minq, float(np.min((Tt - St)[alive])) if alive.any() else minq)
            sg.append(P * (mu - MUC) / (D_EFF / 2) ** (2.0 / 3.0))
            af.append(alive.mean())
        # freeze dead paths so they cannot re-enter or blow up
        Tt = np.where(alive, Tt, TC)
        St = np.where(alive, St, SC)
    return np.array(sg), np.array(af), minq


def run1d(sigma, v, s_start, y_esc, dt, npath, rng, ncheck=400):
    """Reduced normal form dx = (r + x^2) dt + sqrt(D_eff) dW, same ramp."""
    D = (P * sigma) ** 2
    eps = P * v
    r0 = -s_start * (D / 2) ** (2.0 / 3.0)
    ell = (D / 2) ** (1.0 / 3.0)
    x = np.full(npath, -np.sqrt(-r0))
    nstep = int(abs(r0) / eps / dt)
    xesc = y_esc * ell
    alive = np.ones(npath, dtype=bool)
    every = max(1, nstep // ncheck)
    sg, af = [], []
    sq = np.sqrt(D * dt)
    for k in range(nstep):
        r = r0 + eps * (k * dt)
        x = x + (r + x * x) * dt + sq * rng.standard_normal(npath)
        alive &= (x < xesc)
        x = np.where(alive, x, -np.sqrt(max(-r, 1e-12)))
        if k % every == 0:
            sg.append(r / (D / 2) ** (2.0 / 3.0))
            af.append(alive.mean())
    return np.array(sg), np.array(af)


def cross(sg, af, level=0.5):
    """|s| where survival crosses `level` (sg is increasing toward 0)."""
    j = np.argmax(af < level)
    if j == 0 or af[-1] >= level:
        return np.nan
    s0, s1 = sg[j - 1], sg[j]
    p0, p1 = af[j - 1], af[j]
    return abs(s0 + (level - p0) * (s1 - s0) / (p1 - p0))


DT = 0.01
NPATH = 3000
NBATCH = 6

print(f"  dt={DT}, {NBATCH} independent batches x {NPATH} paths, "
      f"start |s|={S_START}, Y_esc={Y_ESC}\n")
print(f"  {'SIG':>7} {'|s_h| closed':>13} {'|s_h| 1-D':>18} {'|s_h| 2-D':>18} "
      f"{'2D/cf':>7}")

rows = []
for SIG in [0.05, 0.02]:
    sigma, v, eps_eff = design(D_EFF, SIG)
    s_cf, r_cf = horizon_cf(D_EFF, eps_eff)
    h1, h2, minqs = [], [], []
    for ib in range(NBATCH):
        rng = np.random.default_rng(SEEDS[ib])
        sg, af = run1d(sigma, v, S_START, Y_ESC, DT, NPATH, rng)
        h1.append(cross(sg, af))
        rng = np.random.default_rng(SEEDS[ib])
        sg2, af2, mq = run2d(sigma, v, S_START, Y_ESC, DT, NPATH, rng)
        h2.append(cross(sg2, af2))
        minqs.append(mq)
    h1 = np.array(h1); h2 = np.array(h2)
    m1, e1 = h1.mean(), h1.std(ddof=1) / np.sqrt(NBATCH)
    m2, e2 = h2.mean(), h2.std(ddof=1) / np.sqrt(NBATCH)
    print(f"  {SIG:7.3f} {s_cf:13.4f} {m1:11.4f} +- {e1:.4f} "
          f"{m2:11.4f} +- {e2:.4f} {m2/s_cf:7.4f}")
    rows.append((SIG, sigma, v, eps_eff, s_cf, r_cf, m1, e1, m2, e2, min(minqs)))

print()
check("q stayed positive (smooth branch valid)", min(r[10] for r in rows) > 0,
      f"min q = {min(r[10] for r in rows):.4f}")
for (SIG, sig, v, eps, s_cf, r_cf, m1, e1, m2, e2, mq) in rows:
    check(f"SIG={SIG}: 2-D horizon within 8% of closed form",
          abs(m2 / s_cf - 1) < 0.08, f"ratio {m2/s_cf:.4f}")
    check(f"SIG={SIG}: 1-D reduction agrees with 2-D within 5%",
          abs(m2 / m1 - 1) < 0.05, f"ratio {m2/m1:.4f}")

# ============================================================== STEP 6
hdr("STEP 6  Robustness: dt, escape threshold, ramp start")
SIG = 0.05
sigma, v, eps_eff = design(D_EFF, SIG)
s_cf, _ = horizon_cf(D_EFF, eps_eff)
rng = np.random.default_rng(SEEDS[7])

print(f"  {'variant':>26} {'|s_h| (2-D)':>13}")
base = None
for lab, kw in [("baseline", dict(dt=DT, y_esc=Y_ESC, s0=S_START)),
                ("dt / 2", dict(dt=DT / 2, y_esc=Y_ESC, s0=S_START)),
                ("Y_esc = 5", dict(dt=DT, y_esc=5.0, s0=S_START)),
                ("Y_esc = 12", dict(dt=DT, y_esc=12.0, s0=S_START)),
                ("start |s| = 4.5", dict(dt=DT, y_esc=Y_ESC, s0=4.5))]:
    rng = np.random.default_rng(SEEDS[7])
    sg, af, _ = run2d(sigma, v, kw["s0"], kw["y_esc"], kw["dt"], 4000, rng)
    h = cross(sg, af)
    if base is None:
        base = h
    print(f"  {lab:>26} {h:13.4f}    ({h/base:.4f} x baseline)")
    if lab != "baseline":
        check(f"insensitive to {lab}", abs(h / base - 1) < 0.05,
              f"ratio {h/base:.4f}")

# ============================================================== SUMMARY
hdr("SUMMARY")
print("  Physical prediction, no fitted parameters:")
for (SIG, sig, v, eps, s_cf, r_cf, m1, e1, m2, e2, mq) in rows:
    dmu = r_cf / P
    print(f"    SIG={SIG:.3f}  sigma={sig:.5f}  v={v:.3e}")
    print(f"        predicted horizon  eta2 = mu_c - {dmu:.6f} = {MUC-dmu:.6f}")
    print(f"        measured  (2-D)    eta2 = mu_c - {m2*(D_EFF/2)**(2/3)/P:.6f} "
          f"= {MUC - m2*(D_EFF/2)**(2/3)/P:.6f}")
print()
if FAIL:
    print(f"  {len(FAIL)} CHECK(S) FAILED:")
    for f_ in FAIL:
        print("    -", f_)
else:
    print("  All checks passed.")
