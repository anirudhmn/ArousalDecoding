"""Out-of-sample validation on the full-BCI condition.

The optimal trajectory, the sensitivity split and the band multipliers are all
derived from hard-course trials under conditions 1 and 2. They are then applied
unchanged to the hard-course condition-3 trials, which never entered any part of
the fit.

The evaluation trials are a different feedback condition from the fitting
trials, so nothing about the fit can carry over through shared data.

Outputs: results/heldout_full_bci_universal.csv,
         results/heldout_full_bci_personalised.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress

from common import C, OUT, cfg, hard_control, trial_table

SENS_MULT, TOL_MULT = 1.5, 2.5


def run():
    df = trial_table()
    fit = hard_control(df)                                   # cond 1+2, hard
    test = df[(df.difficulty == 1)
              & (df.condition == cfg.FEEDBACK_CONDITION)].reset_index(drop=True)

    print("=" * 74)
    print("Held-out validation on condition 3 (full BCI)")
    print("=" * 74)
    print(f"fitting trials  (cond 1+2, hard): {len(fit)}")
    print(f"held-out trials (cond 3,   hard): {len(test)}")

    # Everything fitted on cond 1+2 only.
    traj = C.optimal_trajectory(C.performance_surface(fit))
    score = -fit.hrv_z.to_numpy() + fit.gamma_z.to_numpy()
    median = np.median(score)

    # Group membership is a per-subject property, carried over to the test set.
    sens_subj = set(fit.loc[score > median, "subject"].unique())
    tol_subj = set(fit.subject.unique()) - sens_subj
    overlap = sens_subj & tol_subj
    print(f"sensitive subjects: {len(sens_subj)}   tolerant: {len(tol_subj)}"
          + (f"   (overlap {len(overlap)} - trial-level split is not clean)"
             if overlap else ""))

    s_i = np.where(test.subject.isin(sens_subj))[0]
    t_i = np.where(~test.subject.isin(sens_subj))[0]

    univ = C.deviation_metrics(test, traj, 1.0)
    pers = pd.concat([C.deviation_metrics(test, traj, SENS_MULT, s_i),
                      C.deviation_metrics(test, traj, TOL_MULT, t_i)],
                     ignore_index=True)

    print("\n" + "-" * 74)
    print("Deviation -> performance on the held-out condition-3 trials")
    print("-" * 74)
    for label, frame in [("universal (x1.0)", univ),
                         (f"personalised (x{SENS_MULT}/x{TOL_MULT})", pers)]:
        d, dp = C.cohens_d_good_bad(frame)
        r = linregress(frame.pct_in_band, frame.performance)
        print(f"  {label:<28} n={len(frame):3d}  d={d:5.2f} (p={dp:.3f})  "
              f"r={r.rvalue:+.3f} (p={r.pvalue:.4f})")

    # Same metrics on the fitting set, for reference.
    print("\nfor reference, on the fitting trials (in-sample):")
    u_fit = C.deviation_metrics(fit, traj, 1.0)
    r_fit = linregress(u_fit.pct_in_band, u_fit.performance)
    print(f"  universal (x1.0)             n={len(u_fit):3d}  "
          f"d={C.cohens_d_good_bad(u_fit)[0]:5.2f}          r={r_fit.rvalue:+.3f}")

    univ.to_csv(OUT / "heldout_full_bci_universal.csv", index=False)
    pers.to_csv(OUT / "heldout_full_bci_personalised.csv", index=False)
    return univ, pers


if __name__ == "__main__":
    run()
