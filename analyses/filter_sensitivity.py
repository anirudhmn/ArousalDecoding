"""Does the choice of smoothing filter change any conclusion?

The decoded index is smoothed with an asymmetric first-order IIR filter, faster
on the way up than on the way down, chosen to match sympathetic activation
against parasympathetic recovery rather than tuned on the data. That choice is
a free parameter and the manuscript should not rest on it.

The filter is invertible. Each stored trace holds one value per 16-sample
update, and the step direction reveals which time constant was applied at that
update, so the pre-filter sequence can be recovered exactly and re-smoothed
with any other constants. No decoder is retrained here.

Three symmetric alternatives are compared against the published asymmetric
filter, together with no smoothing at all:

  * the trajectory of decoded arousal over the trial,
  * the deviation metrics and their association with flight time,
  * the trial-level quadratic that carries the inverted-U.

Outputs: results/filter_sensitivity.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

from common import C as ctrl
from common import OUT, hard_control, trial_table
from arousal.signals import asymmetric_iir

UPDATE = 16
WARMUP = 256

# (label, tau_rise, tau_fall). The first row is what the manuscript reports.
FILTERS = [
    ("asymmetric 0.75 / 1.5 (published)", 0.75, 1.5),
    ("symmetric 1.0", 1.0, 1.0),
    ("symmetric 1.5", 1.5, 1.5),
    ("symmetric 0.75", 0.75, 0.75),
    ("no smoothing", None, None),
]


def _updates(trace):
    """The one-value-per-update sequence held inside a sample-rate trace."""
    idx = np.arange(WARMUP, len(trace), UPDATE)
    return idx, trace[idx].astype(float)


def invert(trace):
    """Recover the pre-filter values from a smoothed trace.

    ``S[j] = (1 - a) S[j-1] + a A[j]`` with ``a = a_up`` when ``A[j] > S[j-1]``
    and ``a_down`` otherwise. A rise in S implies the first case and a fall or
    a hold implies the second, so the correct constant is known at every step
    and the inversion is exact up to floating point.
    """
    idx, S = _updates(trace)
    a_up, a_down = asymmetric_iir(UPDATE)
    A = np.empty_like(S)
    prev = 0.0
    for j, s in enumerate(S):
        a = a_up if s > prev else a_down
        A[j] = prev + (s - prev) / a
        prev = s
    return idx, np.clip(A, 0, 100)


def resmooth(idx, A, n, tau_rise, tau_fall):
    """Re-apply a filter to a recovered sequence and expand to sample rate."""
    out = np.zeros(n)
    if tau_rise is None:
        S = A
    else:
        a_up, a_down = asymmetric_iir(UPDATE, tau_rise, tau_fall)
        S = np.empty_like(A)
        prev = 0.0
        for j, a_val in enumerate(A):
            a = a_up if a_val > prev else a_down
            prev = (1 - a) * prev + a * a_val
            S[j] = prev
    for j, i in enumerate(idx):
        out[i:min(i + UPDATE, n)] = S[j]
    return out


def run():
    df = trial_table()

    print("=" * 80)
    print("Sensitivity of every downstream result to the smoothing filter")
    print("=" * 80)

    recovered = [invert(t) for t in df.new_arousal]

    # Inversion check on the published constants: re-smoothing the recovered
    # sequence with the same filter has to return the original trace.
    check = [np.max(np.abs(resmooth(i, a, len(t), 0.75, 1.5) - t))
             for (i, a), t in zip(recovered, df.new_arousal)]
    print(f"\ninversion check: largest deviation after a round trip "
          f"= {max(check):.2e} arousal units")

    rows = []
    for label, tr, tf in FILTERS:
        d = df.copy()
        d["new_arousal"] = [resmooth(i, a, len(t), tr, tf)
                            for (i, a), t in zip(recovered, d.new_arousal)]
        d["arousal"] = [np.mean(t[WARMUP:]) for t in d.new_arousal]

        hc = hard_control(d)
        traj = ctrl.optimal_trajectory(ctrl.performance_surface(hc))
        met = ctrl.deviation_metrics(hc, traj, 1.0)
        r_above = pearsonr(met.pct_above, met.performance)[0]
        r_in = pearsonr(met.pct_in_band, met.performance)[0]
        cd = ctrl.cohens_d_good_bad(met)[0]

        m = smf.mixedlm("performance ~ arousal + I(arousal**2)"
                        " + C(difficulty) + C(condition)", d,
                        groups=d["subject"]).fit(reml=False)
        b2 = m.params["I(arousal ** 2)"]
        b1 = m.params["arousal"]
        opt = -b1 / (2 * b2) if b2 < 0 else np.nan

        rows.append(dict(filter=label, mean_arousal=d.arousal.mean(),
                         r_pct_above=r_above, r_pct_in_band=r_in, cohens_d=cd,
                         beta_quad=b2, p_quad=m.pvalues["I(arousal ** 2)"],
                         optimum=opt))
        print(f"\n{label}")
        print(f"  mean trial arousal      {d.arousal.mean():6.2f}")
        print(f"  r(% above band, flight) {r_above:+6.3f}    "
              f"r(% in band, flight) {r_in:+6.3f}    Cohen d {cd:5.2f}")
        print(f"  trial quadratic         {b2:+6.3f} (p = {m.pvalues['I(arousal ** 2)']:.4f})"
              f"    optimum {opt:5.1f}")

    out = pd.DataFrame(rows)
    print("\n" + "-" * 80)
    print("Reading. The published filter is one row among several. If the sign,")
    print("the significance and the rough size of each quantity hold across the")
    print("rows, the smoothing choice is not carrying any conclusion.")
    print("\nThe first row differs slightly from the headline quadratic in")
    print("duration_permutation.py because trial-mean arousal is recomputed")
    print("here from the resmoothed trace, dropping the warm-up. Every row uses")
    print("that same definition, so the comparison across rows is like for like.")
    out.to_csv(OUT / "filter_sensitivity.csv", index=False)
    print(f"\nwrote {OUT / 'filter_sensitivity.csv'}")


if __name__ == "__main__":
    run()
