#!/usr/bin/env python3
"""
Kibble-Zurek freeze-out vs the observability horizon.

Kibble-Zurek (KZ) scaling is the standard framework for a system ramped
through a critical point, and this checks whether it is ever reached at a
noisy saddle-node before noise-induced escape removes the ensemble.

Setup (scaled normal form). dx/dt = -sigma + x^2, stable point x_- =
-sqrt(sigma), relaxation rate lambda_r = |V''(x_-)| = 2 sqrt(sigma), so

    tau_relax = 1/(2 sqrt(sigma)).

Ramp d sigma/dt = -SIG. Time remaining to threshold t_rem = sigma/SIG.

Kibble-Zurek freeze-out: adiabaticity fails when the relaxation time equals
the time remaining, tau_relax = t_rem:

    1/(2 sqrt(sigma)) = sigma/SIG   =>   sigma_KZ = (SIG/2)^{2/3}.

Observability horizon: noise-induced escape has removed half the ensemble by

    sigma_h = [(3/4) ln(1/(2 pi ln2 SIG))]^{2/3}.

The question: which comes first as the system is ramped in? If sigma_h >
sigma_KZ the ensemble tips while still adiabatic, the quasi-static description
is self-consistent, and KZ physics is never reached. If the reverse, the
quasi-static hazard is invalid before escape matters.

Prediction. sigma_h/sigma_KZ = [(3/2) ln(1/(2 pi ln2 SIG)) / SIG]^{2/3} -> infinity
as SIG -> 0: a power of SIG beaten by a logarithm. The horizon always wins,
and by a growing margin as the ramp slows.
"""
import numpy as np


def sigma_KZ(SIG):
    return (SIG / 2.0) ** (2.0 / 3.0)


def sigma_h(SIG):
    """Closed-form scaled horizon |s_h|.

    This is the scaled quantity, so it carries no D, and it carries the ln 2
    of the median convention.  sigma_KZ below is likewise scaled, so the two
    are directly comparable.
    """
    arg = 1.0 / (2 * np.pi * np.log(2) * SIG)
    if arg <= 1:
        return np.nan
    return (0.75 * np.log(arg)) ** (2.0 / 3.0)


print("=" * 78)
print("KIBBLE-ZUREK FREEZE-OUT vs OBSERVABILITY HORIZON")
print("=" * 78)
print("  sigma_KZ = (SIG/2)^{2/3}          [adiabaticity fails]")
print("  sigma_h  = [(3/4) ln(1/(2 pi ln2 SIG))]^{2/3}  [half the ensemble gone]\n")

print("  Both quantities are scaled, so the comparison is D-independent.\n")
print(f"{'SIG':>10} {'sigma_KZ':>10} {'sigma_h':>10} "
      f"{'ratio h/KZ':>11} {'verdict':>22}")
for SIG in [2.222e-1, 2.222e-2, 2.222e-3, 2.222e-4, 2.222e-5, 2.222e-6, 2.222e-8]:
    kz = sigma_KZ(SIG)
    sh = sigma_h(SIG)
    v = "horizon first (adiabatic)" if sh > kz else "KZ first (non-adiabatic)"
    print(f"{SIG:10.3e} {kz:10.5f} {sh:10.5f} {sh / kz:11.1f}   {v:>22}")

print("\n" + "=" * 78)
print("WHERE, IF ANYWHERE, DOES KZ WIN?")
print("=" * 78)
print("  Solve sigma_h(SIG) = sigma_KZ(SIG) for the crossover SIG.\n")
from scipy.optimize import brentq
xc = brentq(lambda S: sigma_h(S) - sigma_KZ(S), 1e-12,
            1.0 / (2 * np.pi * np.log(2)) * 0.999999)
print(f"  crossover at SIG = {xc:.4f}")
print("  Above this, freeze-out precedes the horizon.  It is a single number:")
print("  both sides are scaled, so it does not depend on D.")

print("\n" + "=" * 78)
print("ASYMPTOTIC STATEMENT")
print("=" * 78)
print("""  sigma_h/sigma_KZ = [ (3/2) ln(1/(2 pi ln2 SIG)) / SIG ]^{2/3}

  A logarithm over a power: the ratio DIVERGES as SIG -> 0. The ensemble is
  always destroyed by noise-induced escape long before the ramp outruns the
  relaxation. Consequences:

  1. The quasi-static (adiabatic) hazard is self-consistent for a stated
     reason rather than by assumption. This is a stronger justification than
     an adiabatic-lag estimate alone, which bounds the correction but does
     not show the KZ regime is unreachable.
  2. Kibble-Zurek scaling is never observed in this system. Any attempt to
     detect KZ freeze-out at a noisy saddle-node measures escape statistics
     instead -- a falsifiable prediction, and a caution for contexts where
     KZ exponents are routinely fitted.
  3. The two 2/3 exponents are a coincidence of the fold's barrier scaling,
     not the same physics: (SIG)^{2/3} is deterministic sweep, [ln(1/SIG)]^{2/3}
     is activated escape. Worth stating explicitly, because they look
     identical.""")
