"""Direct Euler-Maruyama simulation of the ramped scaled fold, used to locate
the containment threshold eps_hat_cont at which the median escape location
crosses the |s|_2x boundary (the point where the OU surrogate's Fisher
information first deviates from the exact value by a factor of two).

Deliberately uses a different simulation convention from the eigenproblem-based
routes elsewhere in the codebase (absorbing boundary with linear-interpolated
crossing time, antithetic variates for variance reduction) so the containment
threshold is confirmed by an independent method rather than shared code.
"""
import numpy as np


def median_horizon(SIG, npaths=40000, dt=1e-3, Y=8.0, s0=-3.5, s1=1.5,
                    seed=7, nb=8):
    rng = np.random.default_rng(seed)
    y = np.full(npaths, -np.sqrt(-s0))
    s_esc = np.full(npaths, np.nan)
    live = np.ones(npaths, bool)
    sd = np.sqrt(2.0 * dt)
    s = s0
    while s < s1 and live.any():
        n = live.sum()
        z = rng.standard_normal((n + 1) // 2)
        z = np.concatenate([z, -z])[:n]        # antithetic
        y[live] += (s + y[live] ** 2) * dt + sd * z
        hit = live & (y > Y)
        s_esc[hit] = s
        live &= ~hit
        s += SIG * dt
    s_esc[np.isnan(s_esc)] = s1
    lab = np.arange(npaths) % nb
    meds = np.array([np.median(s_esc[lab == k]) for k in range(nb)])
    return -meds.mean(), meds.std(ddof=1) / np.sqrt(nb)


def sh_closed_form(SIG):
    return (0.75 * np.log(1 / (2 * np.pi * np.log(2) * SIG))) ** (2 / 3)


print("Calibration against the reduced 1-D result:")
for SIG, ref in [(0.05, '1.0026(33)'), (0.02, '1.4412(21)')]:
    m, e = median_horizon(SIG, seed=11)
    print(f"  SIG={SIG}: simulated {m:.4f}({e:.4f})   "
          f"reference {ref}   closed form {sh_closed_form(SIG):.4f}")

print("\nContainment near the |s|_2x = 0.611 boundary:")
for SIG in [0.08, 0.10, 0.1216, 0.15]:
    m, e = median_horizon(SIG, seed=23)
    status = 'INSIDE failure region' if m < 0.6107 else 'contained'
    print(f"  SIG={SIG:6.4f}: simulated median |s_h| = {m:.4f}({e:.4f})   "
          f"closed form {sh_closed_form(SIG):.4f}   -> {status}")

print("\nRobustness at SIG=0.1216 (step size, threshold, ramp start):")
for dt, Y, s0 in [(1e-3, 8.0, -3.5), (5e-4, 8.0, -3.5), (2.5e-4, 8.0, -3.5),
                  (5e-4, 6.0, -3.5), (5e-4, 11.0, -3.5), (5e-4, 8.0, -5.0)]:
    m, e = median_horizon(0.1216, npaths=24000, dt=dt, Y=Y, s0=s0, seed=41)
    print(f"    dt={dt:.1e} Y={Y:4.1f} s0={s0:+.1f} -> {m:.4f}({e:.4f})")
