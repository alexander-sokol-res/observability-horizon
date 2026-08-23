#!/usr/bin/env python3
"""
generate_stommel_figure.py — Fig. 4: the two-box thermohaline application.

  (a) Survival along the ramp.  Closed form (no fitted parameters) against the
      reduced 1-D normal form and the FULL 2-D Stommel model, at two ramp rates.
  (b) Why it works, and what it says.  The exact equilibrium branch of the
      two-box model, the normal-form parabola fitted to it by centre-manifold
      reduction (not by fitting), and the observability horizon -- which sits a
      visible distance before the fold itself.

Self-contained; mirrors stommel_horizon.py.  Deterministic seeds.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

OK = {"black": "#1a1a1a", "blue": "#0072B2", "vermil": "#D55E00",
      "green": "#009E73", "purple": "#CC79A7", "orange": "#E69F00",
      "gray": "#7f7f7f"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.labelsize": 9.5, "axes.titlesize": 9.0,
    "legend.fontsize": 6.6, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3, "lines.markersize": 3.2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

COL = 3.375
OUT = "figures"
ETA1, ETA3 = 3.0, 0.3
LN2 = np.log(2.0)

# ------------------------------------------------------------------ model
def eta2_of_q(q):
    return (ETA1 * (ETA3 + q) - q * (1 + q) * (ETA3 + q)) / (1 + q)


def Jac(T, S):
    return np.array([[-1 - 2 * T + S, T], [-S, -ETA3 - T + 2 * S]])


_d = lambda q: (eta2_of_q(q + 1e-7) - eta2_of_q(q - 1e-7)) / 2e-7
QC = brentq(_d, 0.1, 0.9, xtol=1e-15, rtol=8.9e-16)
MUC = eta2_of_q(QC)
TC, SC = ETA1 / (1 + QC), MUC / (ETA3 + QC)
J = Jac(TC, SC)

w, V = np.linalg.eig(J)
k = int(np.argmin(np.abs(w)))
E = np.real(V[:, k])
wl, Vl = np.linalg.eig(J.T)
kl = int(np.argmin(np.abs(wl)))
FL = np.real(Vl[:, kl]);  FL = FL / np.dot(FL, E)
A = FL[1]
B = (E[1] - E[0]) * (FL[0] * E[0] + FL[1] * E[1])
P = A * B

D_EFF = 1.0e-4
ELL = (D_EFF / 2) ** (1.0 / 3.0)
SIGMA = np.sqrt(D_EFF) / P
print(f"  mu_c={MUC:.9f}  P={P:.8f}  sigma={SIGMA:.5f}  ell={ELL:.5f}")


def horizon_cf(SIG):
    return ((3.0 / 4.0) * np.log(1.0 / (2 * np.pi * LN2 * SIG))) ** (2.0 / 3.0)


# ------------------------------------------------------------------ runs
def run2d(sigma, v, s_start, y_esc, dt, npath, rng, ncheck=300):
    r0 = -s_start * (D_EFF / 2) ** (2.0 / 3.0)
    mu0 = MUC + r0 / P
    q0 = brentq(lambda q: eta2_of_q(q) - mu0, QC + 1e-14, 3.0, xtol=1e-15)
    Tt = np.full(npath, ETA1 / (1 + q0))
    St = np.full(npath, mu0 / (ETA3 + q0))
    nstep = int(abs(r0) / (P * v) / dt)
    xesc = y_esc * ELL
    alive = np.ones(npath, dtype=bool)
    every = max(1, nstep // ncheck)
    sg, af = [], []
    sq = sigma * np.sqrt(dt)
    for i in range(nstep):
        mu = mu0 + v * (i * dt)
        q = Tt - St
        Tt = Tt + (ETA1 - Tt * (1.0 + q)) * dt
        St = St + (mu - St * (ETA3 + q)) * dt + sq * rng.standard_normal(npath)
        x = B * (FL[0] * (Tt - TC) + FL[1] * (St - SC))
        alive &= (x < xesc)
        if i % every == 0:
            sg.append(abs(P * (mu - MUC)) / (D_EFF / 2) ** (2.0 / 3.0))
            af.append(alive.mean())
        Tt = np.where(alive, Tt, TC)
        St = np.where(alive, St, SC)
    return np.array(sg), np.array(af)


def run1d(sigma, v, s_start, y_esc, dt, npath, rng, ncheck=300):
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
    for i in range(nstep):
        r = r0 + eps * (i * dt)
        x = x + (r + x * x) * dt + sq * rng.standard_normal(npath)
        alive &= (x < xesc)
        x = np.where(alive, x, -np.sqrt(max(-r, 1e-12)))
        if i % every == 0:
            sg.append(abs(r) / (D / 2) ** (2.0 / 3.0))
            af.append(alive.mean())
    return np.array(sg), np.array(af)


# ================================================================== figure
SEEDS = np.random.SeedSequence(20260814).spawn(4)
DT, NPATH, S_START, Y_ESC = 0.01, 5000, 3.5, 8.0
CFG = [(0.050, OK["vermil"]), (0.020, OK["blue"])]

fig, axes = plt.subplots(1, 2, figsize=(COL * 2.05, COL * 0.80))

# ---------------------------------------------------------------- panel (a)
ax = axes[0]
CACHE = "figures/.stommel_curves.npz"
cache = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}

for j, (SIG, c) in enumerate(CFG):
    eps_eff = SIG * D_EFF / 2.0
    v = eps_eff / P
    key = f"{SIG:g}"
    if f"s1_{key}" in cache:
        s1, p1 = cache[f"s1_{key}"], cache[f"p1_{key}"]
        s2, p2 = cache[f"s2_{key}"], cache[f"p2_{key}"]
        print(f"  SIG={SIG}: loaded from cache")
    else:
        rng = np.random.default_rng(SEEDS[j])
        s1, p1 = run1d(SIGMA, v, S_START, Y_ESC, DT, NPATH, rng)
        rng = np.random.default_rng(SEEDS[j])
        s2, p2 = run2d(SIGMA, v, S_START, Y_ESC, DT, NPATH, rng)
        cache.update({f"s1_{key}": s1, f"p1_{key}": p1,
                      f"s2_{key}": s2, f"p2_{key}": p2})
        os.makedirs(OUT, exist_ok=True)
        np.savez(CACHE, **cache)
    sc = np.linspace(3.5, 0.05, 400)
    Pcf = np.exp(-np.exp(-(4.0 / 3.0) * sc ** 1.5) / (2 * np.pi * SIG))
    ax.plot(sc, Pcf, '-', color=c, lw=1.5, zorder=2,
            label=rf"closed form, $\hat{{\varepsilon}}={SIG:g}$")
    ax.plot(s1, p1, '--', color=OK["black"], lw=0.9, alpha=0.85, zorder=3,
            label="reduced 1-D" if j == 0 else None)
    ax.plot(s2[::9], p2[::9], 'o', color=c, ms=3.0, mfc="white", mew=0.8,
            zorder=4, label="full 2-D Stommel" if j == 0 else None)
    print(f"  SIG={SIG}: closed |s_h|={horizon_cf(SIG):.4f}")
ax.axhline(0.5, color=OK["gray"], ls=":", lw=0.9)
ax.text(3.42, 0.53, r"$P=\frac{1}{2}$: horizon", fontsize=6.6,
        color=OK["gray"], ha="left", va="bottom")
ax.set_xlim(3.5, 0.05)
ax.set_ylim(-0.03, 1.06)
ax.set_xlabel(r"$|s| = |r|/(D_{\mathrm{eff}}/2)^{2/3}$   (threshold $\rightarrow$)")
ax.set_ylabel(r"survival probability $P$")
ax.set_title("(a) prediction vs. the full two-dimensional model",
             fontsize=8.2, loc="left")
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.legend(loc="lower left", handlelength=1.9, borderaxespad=0.4, ncol=1)

# ---------------------------------------------------------------- panel (b)
ax = axes[1]
qs_st = np.linspace(QC, 1.35, 400)          # stable branch  (q > q_c)
qs_un = np.linspace(0.02, QC, 400)          # unstable branch (q < q_c)
ax.plot(eta2_of_q(qs_st), qs_st, '-', color=OK["black"], lw=1.5,
        label="stable branch")
ax.plot(eta2_of_q(qs_un), qs_un, '-', color=OK["black"], lw=1.0, alpha=0.4,
        dashes=(4, 2), label="unstable branch")
# normal-form parabola from the centre-manifold reduction (NOT fitted)
dmu = np.linspace(-0.032, 0.0, 300)
r = P * dmu
dq = np.sqrt(-r) / abs(B) * abs(E[1] - E[0])
ax.plot(MUC + dmu, QC + dq, ':', color=OK["green"], lw=1.9, zorder=5,
        label="normal form (reduced)")
ax.plot(MUC + dmu, QC - dq, ':', color=OK["green"], lw=1.9, zorder=5)

# observability horizon at SIG = 0.02
SIGh = 0.020
rh = horizon_cf(SIGh) * (D_EFF / 2) ** (2.0 / 3.0)
eta_h = MUC - rh / P
ax.axvspan(eta_h, MUC, color=OK["gray"], alpha=0.20, lw=0, zorder=1)
ax.axvline(eta_h, color=OK["blue"], lw=1.2, ls="-", zorder=4)

ax.plot([MUC], [QC], marker='o', ms=5.5, color=OK["black"], mfc=OK["orange"],
        mew=0.9, zorder=7, clip_on=False)
ax.annotate("fold", xy=(MUC, QC), xytext=(MUC - 0.0045, QC - 0.135),
            fontsize=7.0, ha="center", zorder=7,
            arrowprops=dict(arrowstyle="->", lw=0.7, color=OK["black"]))

yarr = 0.700
ax.annotate("", xy=(MUC, yarr), xytext=(eta_h, yarr), zorder=7,
            arrowprops=dict(arrowstyle="<->", lw=0.9, color=OK["blue"]))
ax.text((eta_h + MUC) / 2, yarr + 0.018, "never\nobserved", fontsize=6.8,
        color=OK["blue"], ha="center", va="bottom", linespacing=1.15)
ax.text(eta_h - 0.0012, 0.215, "horizon", fontsize=6.9, color=OK["blue"],
        ha="right", va="center", rotation=90)

ax.set_xlim(1.1905, MUC + 0.0022)
ax.set_ylim(0.18, 0.83)
ax.set_xlabel(r"freshwater forcing $\eta_2$   (ramp $\rightarrow$)")
ax.set_ylabel(r"overturning $q=T-S$")
ax.set_title("(b) the reduction, and where observation ends",
             fontsize=8.2, loc="left")
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.legend(loc="upper left", handlelength=1.9, borderaxespad=0.4)

fig.tight_layout(pad=0.4)
os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/fig_stommel.png")
fig.savefig(f"{OUT}/fig_stommel.pdf")
plt.close(fig)
print(f"  fold at eta2={MUC:.6f};  horizon at eta2={eta_h:.6f}"
      f"   (gap {MUC-eta_h:.6f})")
print("  wrote figures/fig_stommel.pdf")
