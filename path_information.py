#!/usr/bin/env python3
"""
Two checks on the continuous-record information rate for the ramped fold:
whether it equals 1/D per unit surviving time uniformly in distance from the
threshold, and whether a discrete-time maximum-likelihood estimator attains
the corresponding Cramer-Rao bound.

Both checks require care about what "surviving" and "observed" mean, since the
natural-looking naive versions introduce selection bias:

  Windowed score variance.  A window's score must be accumulated only over
  paths that are alive at the window's START, using only the information
  available up to that point -- not paths alive at the end of the whole ramp,
  which conditions every window on the entire future and suppresses
  late-window variance purely through survivorship.

  MLE efficiency.  The mean-squared error must be compared against D/T for a
  FIXED observation length T, using only paths observed for the full window --
  not averaged over paths of unequal length, since by Jensen's inequality
  E[D/T_j] >= D/E[T_j] biases the efficiency below 1 through heterogeneity
  alone, independent of any real inefficiency in the estimator.
"""
import numpy as np

D = 2.0
rng = np.random.default_rng(5)

print("=" * 78)
print("Is the information rate 1/D per unit SURVIVING time,")
print("uniformly in distance from the fold?")
print("=" * 78)


def windows(SIG, edges, s0=-2.5, R=40000, dt=1e-3, xesc=8.0):
    n = int(((0.0 - s0) / SIG + 1.0) / dt)
    x = np.full(R, -np.sqrt(-s0))
    alive = np.ones(R, bool)
    nw = len(edges) - 1
    acc = np.zeros((nw, R))         # score accumulated inside window i
    tin = np.zeros((nw, R))         # time spent inside window i
    start = np.zeros((nw, R), bool)  # alive at window start
    prev = -1
    t = 0.0
    for k in range(n):
        if not alive.any():
            break
        s = s0 + SIG * t
        sig = -s
        i = -1
        for j in range(nw):
            if edges[j] >= sig > edges[j + 1]:
                i = j
                break
        if i >= 0 and i != prev:
            start[i] = alive.copy()
            prev = i
        b = s + x[alive] ** 2
        dw = rng.standard_normal(alive.sum()) * np.sqrt(dt)
        dx = b * dt + np.sqrt(D) * dw
        if i >= 0:
            acc[i][alive] += dw / np.sqrt(D)
            tin[i][alive] += dt
        x[alive] += dx
        t += dt
        esc = np.zeros(R, bool)
        esc[alive] = x[alive] > xesc
        alive &= ~esc
        np.clip(x, -1e3, 1e3, out=x)
    return acc, tin, start


SIG = 0.05
edges = [2.5, 2.0, 1.5, 1.0, 0.5, 0.0]
acc, tin, start = windows(SIG, edges)
print(f"  SIG={SIG}, D={D}.  For each window: paths alive at window START,")
print(f"  score accumulated to min(escape, window end).\n")
print(f"{'sigma window':>16} {'n at start':>11} {'E[t in win]':>12} "
      f"{'Var(score)':>11} {'E[t]/D':>9} {'ratio':>8} {'frac surv':>10}")
for i in range(len(edges) - 1):
    sel = start[i]
    if sel.sum() < 50:
        continue
    Et = tin[i][sel].mean()
    v = acc[i][sel].var()
    full = (edges[i] - edges[i + 1]) / SIG
    print(f"  [{edges[i]:.1f},{edges[i + 1]:.1f}) {sel.sum():11d} {Et:12.4f} "
          f"{v:11.5f} {Et / D:9.4f} {v / (Et / D):8.4f} {Et / full:10.4f}")
print("""
  Reading: ratio ~1 in every window means the information rate is 1/D per unit
  SURVIVING time everywhere -- approaching the fold does not raise the rate.
  What falls near the fold is 'frac surv', the expected time you still get.
  So proximity to the bifurcation is informationally NEUTRAL per unit time and
  NEGATIVE in total, because it buys less remaining observation.""")

print("\n" + "=" * 78)
print("Efficiency of an observer sampling at spacing Delta")
print("=" * 78)


def mle_efficiency(SIG, Delta, s0=-2.5, R=20000, dt=1e-3, xesc=8.0, T=10.0):
    n = int(T / dt)
    step = max(1, int(round(Delta / dt)))
    x = np.full(R, -np.sqrt(-s0))
    alive = np.ones(R, bool)
    num = np.zeros(R)
    cnt = np.zeros(R)
    xprev = x.copy()
    sprev = np.full(R, s0)
    t = 0.0
    for k in range(n):
        s = s0 + SIG * t
        b = s + x ** 2
        dx = b * dt + np.sqrt(D) * rng.standard_normal(R) * np.sqrt(dt)
        x = x + dx
        t += dt
        if k % step == step - 1:
            num[alive] += (x[alive] - xprev[alive]) / Delta - \
                (sprev[alive] + xprev[alive] ** 2)
            cnt[alive] += 1
            xprev, sprev = x.copy(), np.full(R, s)
        alive &= (x <= xesc)
        np.clip(x, -1e3, 1e3, out=x)
    full = cnt.max()
    keep = cnt == full                      # observed for the FULL window only
    return num[keep] / full, full * Delta, keep.sum()


print(f"  SIG=0.05, T=10 (fixed), only paths observed for the full window.")
print(f"  Bound is D/T = {D / 10.0:.5f} for every retained path.\n")
print(f"{'Delta':>8} {'N samp':>8} {'n paths':>9} {'MSE':>11} {'D/T':>10} "
      f"{'efficiency':>11}")
for Delta in [0.001, 0.005, 0.02, 0.1, 0.5, 1.0]:
    th, Teff, npath = mle_efficiency(0.05, Delta)
    mse = float((th ** 2).mean())
    bound = D / Teff
    print(f"{Delta:8.3f} {Teff / Delta:8.0f} {npath:9d} {mse:11.5f} "
          f"{bound:10.5f} {bound / mse:11.4f}")
print("""
  Efficiency -> 1 as Delta -> 0 confirms the path bound is attainable, and the
  falloff at large Delta is discretisation bias (the drift is evaluated at the
  left endpoint), not lost information.""")
