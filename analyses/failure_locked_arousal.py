"""Decoded arousal locked to failure events.

A trial ends either at a boundary crash or at the 90 s ceiling. The last seconds
before a crash are extracted and compared against the trial's own earlier
baseline, and then across the three feedback conditions.

This asks whether feedback acts locally, near the moments where performance is
actually lost, rather than by shifting the whole trajectory.

Outputs: results/failure_locked_arousal.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import f_oneway, ttest_rel

from common import C, OUT, cfg, trial_table
from common import FS

PRE_S = 10          # window before the crash
BASE_S = (20, 10)   # baseline window, seconds before the crash
CEILING_S = 88      # trials at least this long reached the time limit


def run():
    df = trial_table()
    df = df[df.difficulty == 1].copy()
    df["dur_s"] = df.performance / FS
    df["crashed"] = df.dur_s < CEILING_S

    print("=" * 74)
    print("Arousal locked to failure (boundary crash)")
    print("=" * 74)
    print(f"hard-course trials: {len(df)}   "
          f"crashed: {df.crashed.sum()}   reached ceiling: {(~df.crashed).sum()}")
    print(f"duration: min {df.dur_s.min():.1f}s  median {df.dur_s.median():.1f}s  "
          f"max {df.dur_s.max():.1f}s")

    rows = []
    for r in df[df.crashed].itertuples():
        tr = r.new_arousal
        if len(tr) < BASE_S[0] * FS:
            continue
        pre = tr[-PRE_S * FS:].mean()
        base = tr[-BASE_S[0] * FS:-BASE_S[1] * FS].mean()
        rows.append({"subject": r.subject, "condition": r.condition,
                     "dur_s": r.dur_s, "pre_crash": pre, "baseline": base,
                     "rise": pre - base})
    fl = pd.DataFrame(rows)
    print(f"\nusable crashed trials (>= {BASE_S[0]}s): {len(fl)}")

    # --- (a) does arousal rise into the crash? -----------------------------
    t, p = ttest_rel(fl.pre_crash, fl.baseline)
    m = smf.mixedlm("rise ~ 1", fl, groups=fl["subject"]).fit(reml=True)
    ci = m.conf_int()
    print("\n" + "-" * 74)
    print(f"arousal in the last {PRE_S}s vs the preceding {BASE_S[0]}-{BASE_S[1]}s window")
    print("-" * 74)
    print(f"  baseline  {fl.baseline.mean():5.1f}   pre-crash {fl.pre_crash.mean():5.1f}   "
          f"rise {fl.rise.mean():+5.1f}")
    print(f"  paired t({len(fl)-1}) = {t:.2f}, p = {p:.2e}")
    print(f"  mixed model intercept = {m.params['Intercept']:+.2f} "
          f"[{ci.loc['Intercept',0]:+.2f}, {ci.loc['Intercept',1]:+.2f}], "
          f"p = {m.pvalues['Intercept']:.2e}")

    # --- (b) does the rise differ by feedback condition? -------------------
    print("\n" + "-" * 74)
    print("by feedback condition")
    print("-" * 74)
    print(f"{'condition':<14}{'n':>4}{'baseline':>10}{'pre-crash':>11}{'rise':>8}")
    groups = []
    for c in (1, 2, 3):
        g = fl[fl.condition == c]
        groups.append(g.rise.to_numpy())
        print(f"{cfg.CONDITIONS[c]:<14}{len(g):>4}{g.baseline.mean():>10.1f}"
              f"{g.pre_crash.mean():>11.1f}{g.rise.mean():>+8.1f}")
    F, p = f_oneway(*groups)
    print(f"\n  one-way ANOVA on rise: F = {F:.2f}, p = {p:.3f}")
    mm = smf.mixedlm("rise ~ C(condition)", fl, groups=fl["subject"]).fit(reml=False)
    m0 = smf.mixedlm("rise ~ 1", fl, groups=fl["subject"]).fit(reml=False)
    from scipy.stats import chi2
    lr = 2 * (mm.llf - m0.llf)
    print(f"  mixed model LRT for condition: chi2(2) = {lr:.2f}, "
          f"p = {chi2.sf(lr, 2):.3f}")

    # --- (c) time course, condition by condition ---------------------------
    print("\n" + "-" * 74)
    print("mean arousal in each second before the crash")
    print("-" * 74)
    span = 15
    print(f"{'s before crash':<16}" + "".join(f"{c:>7}" for c in
                                              ["silence", "half", "bci"]))
    curves = {c: [] for c in (1, 2, 3)}
    for r in df[df.crashed].itertuples():
        tr = r.new_arousal
        if len(tr) < span * FS:
            continue
        binned = [tr[-(k + 1) * FS:len(tr) - k * FS].mean() for k in range(span)]
        curves[r.condition].append(binned)
    for k in range(0, span, 2):
        line = f"{-(k+1):<16}"
        for c in (1, 2, 3):
            arr = np.array(curves[c])
            line += f"{arr[:, k].mean():>7.1f}"
        print(line)

    fl.to_csv(OUT / "failure_locked_arousal.csv", index=False)
    return fl


if __name__ == "__main__":
    run()
