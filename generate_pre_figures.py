#!/usr/bin/env python3
"""
generate_pre_figures.py — figures for the manuscript.

  fig_information.pdf/.png   Fig 1: I_OU diverges as |s|^-2; the exact
                                    conditioned law SATURATES -> D^-4/3 plateau
  fig_conditioned.pdf/.png   Fig 2: the conditioned law -- cutoff convergence,
                                    flux tail, boundary-condition sensitivity
  fig_horizon.pdf/.png       Fig 3: observability horizon and its eps-independence

Deterministic: eigsh is given an explicit start vector (its default v0 is
random, which otherwise makes output vary in the last digits).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import CubicSpline

OK = {"black": "#1a1a1a", "blue": "#0072B2", "vermil": "#D55E00",
      "green": "#009E73", "purple": "#CC79A7", "orange": "#E69F00",
      "gray": "#7f7f7f"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.labelsize": 9.5, "axes.titlesize": 9.0,
    "legend.fontsize": 7.0, "xtick.labelsize": 8, "ytick.labelsize": 8,
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
# Quoted plateau, main.tex eq:plateau.  I_s at s = 0 EXACTLY (not at a small
# negative proxy): fisher_inf(0.0) = 0.487522, stable to 5e-5 across walls
# 40-80, grids 80k-160k and steps ds = 0.005-0.02.
PLATEAU_QUOTED = 0.4875
DSC = 2.0
D0 = 0.09


def savefig(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{stem}.png")
    fig.savefig(f"{OUT}/{stem}.pdf")


def loglog_ticks(ax):
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for A in (ax.xaxis, ax.yaxis):
        A.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1,
                                       numticks=20))
        A.set_minor_formatter(NullFormatter())


# ------------------------------------------------------------------ solver
# Imported from verify_all.py, rather than kept as a separate copy, so the
# figures and the reported numbers cannot drift apart.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_va", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_all.py"))
_va = _ilu.module_from_spec(_spec)
import sys as _sys
_src = open(_spec.origin).read().split("# ------------------------------------------------------------------- A")[0]
exec(compile(_src, "verify_all.py", "exec"), _va.__dict__)
qsd, fisher, fisher_inf, tail_mass = _va.qsd, _va.fisher, _va.fisher_inf, _va.tail_mass


def I_ou(s):
    a = abs(s)
    return 1.0 / (2 * np.sqrt(a)) + 1.0 / (8 * a ** 2)


# ==========================================================================
# Fig 1 — information: OU diverges, the exact law saturates
# ==========================================================================
def fig_information():
    S = -np.logspace(np.log10(8.0), np.log10(8e-3), 18)
    Is, Lam = [], []
    for s in S:
        I, lam = fisher_inf(s)
        Is.append(I); Lam.append(lam)
    Is = np.array(Is)
    # s = 0 EXACTLY for the plateau and for the inset; the log axis cannot show
    # it, but it is the value eq:plateau quotes and the inset is linear.
    I0, _ = fisher_inf(0.0)
    plateau = I0

    fig, ax = plt.subplots(figsize=(COL, COL * 0.86))
    A = np.abs(S)
    ax.plot(A, I_ou(S), color=OK["vermil"], lw=1.4,
            label=r"OU surrogate $\;\propto|s|^{-2}$")
    ax.plot(A, Is, 'o-', ms=3.2, color=OK["blue"], mfc="white", mew=0.8,
            label=r"exact conditioned law")
    ax.axhline(PLATEAU_QUOTED, color=OK["black"], ls="--", lw=0.9)
    # Draw the line at the quoted extrapolated plateau, not at the raw
    # computed value: the figure must agree with eq:plateau in main.tex.
    ax.text(6.5, PLATEAU_QUOTED * 1.25,
            r"$\mathcal{I}_s(0)=0.4875(1)$",
            fontsize=7.0, ha="left", va="bottom")

    # Unreachable region: the median trajectory does not enter |s| < |s|_2x
    # for any ramp rate below the containment threshold SIG_cont, since
    # SIG_cont is defined precisely by |s_h|(SIG_cont) = |s|_2x.
    S_2X = 0.6107          # brentq on I_OU(-s) = 2 I_s(0), I_s(0) = 0.487522
    ax.axvspan(S_2X, A.min() * 0.7, color=OK["gray"], alpha=0.18, lw=0)
    ax.axvline(S_2X, color=OK["black"], ls=":", lw=0.9)
    ax.text(S_2X * 1.30, 0.245, "OU off by $2\\times$", fontsize=6.7,
            ha="right", va="bottom", color=OK["black"])
    ax.text(0.05, 8, "not reached by the\nmedian trajectory\nfor $\\hat{\\varepsilon}<0.10$",
            fontsize=6.6, ha="center", va="center", color=OK["black"])

    # ---- linear-scale inset: the log axes cannot show the peak and the 10%
    # decline, which is the content of eq:plateau.
    axi = ax.inset_axes([0.545, 0.545, 0.43, 0.40], zorder=5)
    axi.set_facecolor("white")
    axi.patch.set_alpha(1.0)
    m = A <= 3.0
    Ain = np.concatenate([A[m], [0.0]])
    Iin = np.concatenate([Is[m], [I0]])
    axi.plot(Ain, Iin, 'o-', ms=2.4, color=OK["blue"], mfc="white", mew=0.7, lw=1.0)
    axi.axhline(PLATEAU_QUOTED, color=OK["black"], ls="--", lw=0.7)
    axi.set_xlim(3.0, 0.0)
    axi.set_ylim(0.44, 0.56)
    axi.set_yticks([0.45, 0.50, 0.55])
    axi.set_xticks([3, 2, 1, 0])
    axi.tick_params(labelsize=5.6, length=2.0, pad=1.4)
    axi.set_xlabel(r"$|s|$", fontsize=6.0, labelpad=0.5)
    axi.set_title(r"$\mathcal{I}_s$, linear scale", fontsize=6.0, pad=1.8)
    for sp in axi.spines.values():
        sp.set_linewidth(0.6)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(8.0, A.min() * 0.7)
    ax.set_ylim(0.2, 2e4)
    ax.set_xlabel(r"$|s| = |r|/(D/2)^{2/3}$  (threshold $\rightarrow$)")
    ax.set_ylabel(r"Fisher information about $r$ (scaled)")
    loglog_ticks(ax)
    ax.legend(loc="upper left", handlelength=1.8, borderaxespad=0.35)
    fig.tight_layout(pad=0.35)
    savefig(fig, "fig_information")
    plt.close(fig)
    print(f"Fig 1 done. plateau I_s(0)={plateau:.4f}")
    return S, Is


# ==========================================================================
# Fig 2 — the conditioned law
# ==========================================================================
def fig_conditioned():
    fig, axes = plt.subplots(1, 2, figsize=(COL * 2.05, COL * 0.78))

    # (a) convergence of lambda in the cutoff + Dirichlet-at-x** comparison
    ax = axes[0]
    yRs = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    for s, c in [(-0.5, OK["blue"]), (-2.0, OK["vermil"])]:
        lams = np.array([qsd(s, y)[2] for y in yRs])
        ax.plot(yRs, lams / lams[-1], 'o-', ms=3.2, color=c, mfc="white",
                mew=0.8, label=rf"$s={s}$")
        # Dirichlet exactly at x** (the naive choice)
        xss = np.sqrt(-s)
        lam_x = qsd(s, xss)[2]
        ax.plot([xss], [lam_x / lams[-1]], marker='*', ms=9, color=c,
                clip_on=False, zorder=5)
    ax.axhline(1.0, color=OK["black"], ls="--", lw=0.8)
    ax.text(4.9, 1.06, "explosion limit", fontsize=6.8, ha="right")
    ax.text(1.55, 1.95, r"$\star$ : killing at $x^{**}$" + "\n(naive choice)",
            fontsize=6.8, va="top", color=OK["black"])
    ax.set_xlabel(r"cutoff $y_R$")
    ax.set_ylabel(r"$\lambda(s,y_R)\,/\,\lambda(s,\infty)$")
    ax.set_title("(a) the killing boundary matters", fontsize=8.2, loc="left")
    ax.set_ylim(0.9, 2.15)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(loc="upper right", handlelength=1.5, borderaxespad=0.35)

    # (b) flux tail
    ax2 = axes[1]
    for s, c in [(-0.5, OK["blue"]), (-2.0, OK["vermil"])]:
        # Wall well beyond the plotted range: at yR = 8 the Dirichlet boundary
        # layer contaminates y > 7, which is why this panel used to stop short of
        # the y = 10 the text cites.
        yi, rho, lam = qsd(s, 30.0)
        m = (yi > 1.2) & (yi < 12.0)
        ax2.plot(yi[m], rho[m] * (s + yi[m] ** 2) / lam, color=c,
                 label=rf"$s={s}$")
    ax2.axhline(1.0, color=OK["black"], ls="--", lw=0.8)
    ax2.set_xlabel(r"$y$")
    ax2.set_ylabel(r"$\rho(y)\,(s+y^2)\,/\,\lambda$")
    ax2.set_title(r"(b) flux tail $\rho\to\lambda/(s+y^2)$",
                  fontsize=8.2, loc="left")
    ax2.set_ylim(0, 1.15)
    ax2.tick_params(which="both", direction="in", top=True, right=True)
    ax2.legend(loc="lower right", handlelength=1.5, borderaxespad=0.35)

    fig.tight_layout(pad=0.4)
    savefig(fig, "fig_conditioned")
    plt.close(fig)
    print("Fig 2 done.")


# ==========================================================================
# Fig 3 — observability horizon
# ==========================================================================
def fig_horizon():
    Sg = -np.concatenate([np.arange(6.0, 1.0, -0.2), np.arange(1.0, 0.04, -0.08)])
    Lam = np.array([max(qsd(s, 4.5)[2], 0.0) for s in Sg])

    fig, ax = plt.subplots(figsize=(COL, COL * 0.86))
    cols = [OK["blue"], OK["green"], OK["vermil"], OK["purple"]]
    # Distinguish the four survival curves by dash pattern as well as hue: the
    # Okabe-Ito palette is colorblind-safe, but in grayscale these four map to
    # 8-bit levels within ~40 of one another and would not separate.
    styles = ["-", "--", "-.", ":"]
    horizons = []
    for (e, c, st) in zip([1e-4, 1e-5, 1e-6, 1e-7], cols, styles):
        H = np.concatenate([[0.0], np.cumsum(
            (D0 / 2 / e) * 0.5 * (Lam[1:] + Lam[:-1]) * np.abs(np.diff(Sg)))])
        P = np.exp(-H)
        ax.plot(np.abs(Sg), P, color=c, lw=1.3, ls=st,
                label=rf"$\varepsilon=10^{{{int(np.log10(e))}}}$")
        # Interpolate the P=1/2 crossing rather than snapping to the first
        # grid point past it; with 0.2 spacing snapping costs up to 0.2 in
        # |s_h| (it reported 2.20 where Table II has 2.275).  Same linear
        # interpolation in H as verify_all.py:186-188, so these values are a
        # cross-check on Table II rather than a competing estimate.
        i = int(np.argmax(P < 0.5))
        if i == 0:
            horizons.append(np.nan)
        else:
            f = (np.log(0.5) + H[i - 1]) / (-(H[i] - H[i - 1]))
            horizons.append(abs(Sg[i - 1]) - f * abs(abs(Sg[i]) - abs(Sg[i - 1])))
    ax.axhline(0.5, color=OK["black"], ls=":", lw=0.8)
    ax.text(6.7, 0.535, "survival $=1/2$", fontsize=6.8, va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_xlim(7.0, 0.9)
    ax.set_ylim(-0.03, 1.26)
    ax.set_xlabel(r"$|s| = |r|/(D/2)^{2/3}$  (threshold $\rightarrow$)")
    ax.set_ylabel("survival probability")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax.xaxis.set_major_locator(FixedLocator([7, 5, 4, 3, 2, 1.5, 1.0]))
    ax.xaxis.set_major_formatter(FixedFormatter(["7", "5", "4", "3", "2", "1.5", "1"]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.legend(loc="lower left", handlelength=1.6, borderaxespad=0.4,
              labelspacing=0.3)
    ax.annotate("", xy=(5.2, 1.10), xytext=(2.6, 1.10),
                arrowprops=dict(arrowstyle="->", lw=0.9, color=OK["gray"]))
    ax.text(3.6, 1.13, "horizon recedes as $\\varepsilon$ decreases",
            fontsize=6.9, ha="center", va="bottom", color=OK["gray"])
    fig.tight_layout(pad=0.35)
    savefig(fig, "fig_horizon")
    plt.close(fig)
    print("Fig 3 done. horizons |s| =", [f"{h:.2f}" for h in horizons])


if __name__ == "__main__":
    fig_conditioned()
    fig_horizon()
    fig_information()
    print("\nAll figures written to", OUT)
