"""Leave-one-subject-out validation of the personalised control bands.

Selecting the band multipliers and evaluating them on the same trials would
overstate their value, so per fold everything is rebuilt from the training
subjects only:

  * the optimal arousal trajectory,
  * the sensitivity median that defines the two groups,
  * the grid-searched band multiplier for each group.

The held-out subject is then assigned to a group by their own score against the
training median, and scored with that group's multiplier. Held-out trials are
pooled across folds, and the effect size and correlation are computed once.

Outputs: results/loso_control_bands_universal.csv,
         results/loso_control_bands_personalised.csv,
         results/loso_control_bands_folds.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

from common import C, D, OUT, hard_control, subject_scores, trial_table


def run():
    df = trial_table()
    hc = hard_control(df)
    scores = subject_scores(df)
    subjects = sorted(hc.subject.unique())

    rows_pers, rows_univ, fold_log = [], [], []

    for test_subj in subjects:
        train = hc[hc.subject != test_subj].reset_index(drop=True)
        test = hc[hc.subject == test_subj].reset_index(drop=True)
        if len(test) == 0:
            continue

        # --- everything below is fitted on training subjects only -----------
        traj = C.optimal_trajectory(C.performance_surface(train))

        train_scores = scores.drop(index=test_subj)
        median = train_scores.median()
        sens_subj = set(train_scores[train_scores > median].index)

        sens_idx = np.where(train.subject.isin(sens_subj))[0]
        tol_idx = np.where(~train.subject.isin(sens_subj))[0]

        def best_mult(idx):
            ds, _ = C.band_width_sweep(train, traj, idx)
            return C.BAND_MULTIPLIERS[int(np.nanargmax(ds))], float(np.nanmax(ds))

        m_sens, d_sens = best_mult(sens_idx)
        m_tol, d_tol = best_mult(tol_idx)

        # --- apply to the held-out subject ----------------------------------
        is_sensitive = scores[test_subj] > median
        m_test = m_sens if is_sensitive else m_tol

        pers = C.deviation_metrics(test, traj, m_test)
        univ = C.deviation_metrics(test, traj, 1.0)
        pers["subject"] = test_subj
        univ["subject"] = test_subj
        rows_pers.append(pers)
        rows_univ.append(univ)

        fold_log.append({
            "subject": test_subj, "n_trials": len(test),
            "group": "sensitive" if is_sensitive else "tolerant",
            "m_applied": m_test, "m_sens_train": m_sens, "m_tol_train": m_tol,
            "d_sens_train": d_sens, "d_tol_train": d_tol,
        })

    pers = pd.concat(rows_pers, ignore_index=True)
    univ = pd.concat(rows_univ, ignore_index=True)
    log = pd.DataFrame(fold_log)

    print("=" * 74)
    print("Leave-one-subject-out validation of personalised control bands")
    print("=" * 74)
    print("\nPer-fold multipliers selected on training subjects:")
    print(log.to_string(index=False, float_format="%.2f"))
    print(f"\nmultiplier stability  sensitive: {sorted(log.m_sens_train.unique())}")
    print(f"                      tolerant : {sorted(log.m_tol_train.unique())}")

    print("\n" + "-" * 74)
    print("Held-out results (all folds pooled)")
    print("-" * 74)
    for label, frame in [("universal (x1.0)", univ), ("personalised (LOSO)", pers)]:
        d, p = C.cohens_d_good_bad(frame)
        r = linregress(frame.pct_in_band, frame.performance)
        print(f"  {label:<22} n={len(frame):3d}  Cohen's d={d:5.2f} (p={p:.3f})  "
              f"r={r.rvalue:+.3f} (p={r.pvalue:.4f})")

    # In-sample reference, fitted on all subjects at once.
    traj_full = C.optimal_trajectory(C.performance_surface(hc))
    tscore = -hc.hrv_z.to_numpy() + hc.gamma_z.to_numpy()
    med = np.median(tscore)
    s_i, t_i = np.where(tscore > med)[0], np.where(tscore <= med)[0]
    ins = pd.concat([C.deviation_metrics(hc, traj_full, 1.5, s_i),
                     C.deviation_metrics(hc, traj_full, 2.5, t_i)], ignore_index=True)
    r_ins = linregress(ins.pct_in_band, ins.performance)
    print(f"  {'in-sample (pooled)':<22} n={len(ins):3d}  "
          f"{'':<21}r={r_ins.rvalue:+.3f}")

    pers.to_csv(OUT / "loso_control_bands_personalised.csv", index=False)
    univ.to_csv(OUT / "loso_control_bands_universal.csv", index=False)
    log.to_csv(OUT / "loso_control_bands_folds.csv", index=False)
    return pers, univ, log


if __name__ == "__main__":
    run()
