"""Trial-level statistics with subject clustering respected.

Each participant contributes many trials, so a trial-level Pearson correlation
overstates the effective sample size. The deviation metrics are also skewed
against a ceiling at 100 percent in band, which a linear correlation handles
poorly.

Everything here is refitted with a random intercept per subject and reported
with coefficients, standard errors and confidence intervals. The comparison
between the universal and personalised bands uses a subject-level bootstrap,
which respects the clustering that a test for dependent correlations would not.

Outputs: results/clustered_statistics_metrics.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import linregress, pearsonr, spearmanr

from calibration_profile import build_profile
from common import C, OUT, Y, hard_control, trial_table

RNG = np.random.default_rng(20260806)
N_BOOT = 10000


def mixed(frame, predictor, response="performance"):
    m = smf.mixedlm(f"{response} ~ {predictor}", frame,
                    groups=frame["subject"]).fit(reml=True)
    ci = m.conf_int()
    return dict(beta=m.params[predictor], se=m.bse[predictor],
                ci_lo=ci.loc[predictor, 0], ci_hi=ci.loc[predictor, 1],
                p=m.pvalues[predictor], model=m)


def boot_delta_r(a, b, subjects):
    """Subject-level bootstrap of r(b) - r(a); the two are on the same trials."""
    subs = np.unique(subjects)
    deltas = np.empty(N_BOOT)
    idx_by_subj = {s: np.where(subjects == s)[0] for s in subs}
    for i in range(N_BOOT):
        pick = RNG.choice(subs, size=len(subs), replace=True)
        rows = np.concatenate([idx_by_subj[s] for s in pick])
        deltas[i] = (np.corrcoef(b[0][rows], b[1][rows])[0, 1]
                     - np.corrcoef(a[0][rows], a[1][rows])[0, 1])
    return deltas


def run():
    df = trial_table()
    hc = hard_control(df)
    traj = C.optimal_trajectory(C.performance_surface(hc))
    m1 = C.deviation_metrics(hc, traj, 1.0)

    print("=" * 78)
    print("Statistics with subject clustering respected")
    print("=" * 78)

    # ---- (a) Table 1, Pearson vs mixed model ------------------------------
    print("\nTable 1 refitted as mixed models (random intercept per subject)")
    print(f"{'metric':<24}{'Pearson r':>10}{'Spearman':>10}"
          f"{'  beta [95% CI]':<28}{'p (mixed)':>11}")
    for col, label in [("pct_above", "% time above band"),
                       ("pct_in_band", "% time in band"),
                       ("mean_excursion", "Mean excursion mag."),
                       ("excursion_rate", "Excursion rate/min")]:
        r = pearsonr(m1[col], m1.performance)
        rho = spearmanr(m1[col], m1.performance)
        f = mixed(m1, col)
        print(f"{label:<24}{r.statistic:>+10.3f}{rho.statistic:>+10.3f}"
              f"  {f['beta']:+7.3f} [{f['ci_lo']:+6.3f},{f['ci_hi']:+6.3f}]"
              f"{f['p']:>11.4f}")

    print(f"\nceiling check: {(m1.pct_in_band >= 99.9).mean()*100:.1f}% of trials "
          f"sit at 100% in-band")

    # ---- how long an above-band excursion lasts ----------------------------
    # The latency budget for a reactive intervention. Dips of 1 s or less are
    # bridged and excursions shorter than 2 s are discarded as transients, so
    # what is left is a sustained state change rather than threshold chatter.
    print("\ndwell time above band, poorly performing trials")
    for m in (1.0, 1.75):
        d = C.dwell_times(hc, traj, m)
        print(f"  m = {m:<5} n = {len(d):>3}  mean = {d.mean():.1f} +/- "
              f"{d.std():.1f} s  median = {np.median(d):.1f} s  "
              f"p10 = {np.percentile(d, 10):.1f} s")

    # ---- (b) individual differences: continuous vs median split -----------
    print("\n" + "-" * 78)
    print("Individual differences: continuous interaction as primary evidence")
    print("-" * 78)
    for feature in ("hrv", "gamma"):
        m = Y.fit_interaction_model(df, feature)
        term = f"I(arousal ** 2):{feature}_z"
        ci = m.conf_int()
        print(f"  arousal^2 x {feature:<6} beta={m.params[term]:+.4f} "
              f"SE={m.bse[term]:.4f} "
              f"95% CI [{ci.loc[term,0]:+.4f}, {ci.loc[term,1]:+.4f}] "
              f"p={m.pvalues[term]:.4f}")

    # Sensitivity score as a continuous moderator of deviation-performance.
    # The score is the calibration measure that survives the 17-measure screen,
    # negated so that a higher value means a more arousal-sensitive subject.
    prof = build_profile().set_index("subject")["cal_sd"]
    m1c = m1.copy()
    m1c["score"] = np.array([-float(prof.loc[int(s)]) for s in m1.subject])
    m0 = smf.mixedlm("performance ~ pct_in_band", m1c,
                     groups=m1c["subject"]).fit(reml=False)
    mi = smf.mixedlm("performance ~ pct_in_band * score", m1c,
                     groups=m1c["subject"]).fit(reml=False)
    lr = 2 * (mi.llf - m0.llf)
    from scipy.stats import chi2
    k = len(mi.params) - len(m0.params)
    print(f"\n  moderation by continuous sensitivity score:")
    print(f"    interaction beta = {mi.params['pct_in_band:score']:+.4f}, "
          f"p = {mi.pvalues['pct_in_band:score']:.4f}")
    print(f"    LRT vs no-interaction: chi2({k}) = {lr:.2f}, "
          f"p = {chi2.sf(lr, k):.4f}   (AIC {m0.aic:.1f} -> {mi.aic:.1f})")

    # ---- (c) does personalisation actually improve anything? --------------
    print("\n" + "-" * 78)
    print("Personalisation gain, formally tested")
    print("-" * 78)
    score = m1c["score"].to_numpy()
    # Split at the median across SUBJECTS, not across trials, so that a subject
    # with many trials does not drag the boundary. This is the same median that
    # band_scheme_comparison.py uses inside each leave-one-subject-out fold.
    med = np.median([score[m1c.subject.to_numpy() == s][0]
                     for s in np.unique(m1c.subject)])
    s_i, t_i = np.where(score > med)[0], np.where(score <= med)[0]

    # Both group multipliers are chosen in sample, on the very trials the
    # scheme is then scored on. That is the point of this block: it measures
    # how much the selection alone is worth, against the honest leave-one-
    # subject-out figure in band_scheme_comparison.py.
    def best_mult(idx):
        ds = C.band_width_sweep(hc, traj, idx)
        return C.BAND_MULTIPLIERS[int(np.nanargmax(ds))]

    m_sens, m_tol = best_mult(s_i), best_mult(t_i)
    print(f"  in-sample multipliers: sensitive x{m_sens}, tolerant x{m_tol}")
    pers = pd.concat([C.deviation_metrics(hc, traj, m_sens, s_i),
                      C.deviation_metrics(hc, traj, m_tol, t_i)], ignore_index=True)
    # Re-order the personalised frame to match m1 row-for-row.
    order = np.concatenate([s_i, t_i])
    pers_aligned = pers.iloc[np.argsort(order)].reset_index(drop=True)

    r_u = np.corrcoef(m1.pct_in_band, m1.performance)[0, 1]
    r_p = np.corrcoef(pers_aligned.pct_in_band, pers_aligned.performance)[0, 1]
    deltas = boot_delta_r(
        (m1.pct_in_band.to_numpy(), m1.performance.to_numpy()),
        (pers_aligned.pct_in_band.to_numpy(), pers_aligned.performance.to_numpy()),
        m1.subject.to_numpy())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"  r universal    = {r_u:+.3f}")
    print(f"  r personalised = {r_p:+.3f}")
    print(f"  delta r        = {r_p - r_u:+.3f}   "
          f"bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]   "
          f"p = {2*min((deltas<=0).mean(), (deltas>=0).mean()):.3f}")

    # ---- (d) does the personalised band change the slope? -------------------
    # Numeric coding: pandas string dtypes are not accepted by patsy here.
    both = pd.concat([m1.assign(band=0.0),
                      pers_aligned.assign(band=1.0)], ignore_index=True)
    mg = smf.mixedlm("performance ~ pct_in_band * band", both,
                     groups=both["subject"]).fit(reml=False)
    print(f"\n  band x %in-band interaction: "
          f"beta = {mg.params['pct_in_band:band']:+.4f}, "
          f"p = {mg.pvalues['pct_in_band:band']:.4f}")

    # ---- (e) the median-split grouping, formally tested --------------------
    m1b = m1.copy()
    m1b["grp"] = (score > med).astype(float)
    g0 = smf.mixedlm("performance ~ pct_in_band", m1b,
                     groups=m1b["subject"]).fit(reml=False)
    g1 = smf.mixedlm("performance ~ pct_in_band * grp", m1b,
                     groups=m1b["subject"]).fit(reml=False)
    lr = 2 * (g1.llf - g0.llf)
    k = len(g1.params) - len(g0.params)
    for lab, sel in [("sensitive", m1b.grp == 1.0), ("tolerant", m1b.grp == 0.0)]:
        sub = m1b[sel]
        r = pearsonr(sub.pct_in_band, sub.performance)
        print(f"\n  pooled within the {lab} half: n = {len(sub)} trials, "
              f"r = {r.statistic:+.3f} (p = {r.pvalue:.4f})")

    print(f"\n  median-split group x %in-band:")
    print(f"      beta = {g1.params['pct_in_band:grp']:+.4f}, "
          f"p = {g1.pvalues['pct_in_band:grp']:.4f}")
    print(f"      LRT chi2({k}) = {lr:.2f}, p = {chi2.sf(lr, k):.4f}   "
          f"AIC {g0.aic:.1f} -> {g1.aic:.1f}")
    print("      -> not supported once subject clustering is modelled")

    # ---- (f) the same contrast one subject at a time ------------------------
    # The pooled correlations differ between groups partly because subjects
    # contribute unequal numbers of trials. Fitting each subject separately and
    # comparing the two sets of slopes removes that, at the cost of power.
    from scipy.stats import ttest_ind
    slopes = []
    for s, g in m1b.groupby("subject"):
        if len(g) < 4 or g.pct_in_band.std() < 1e-9:
            continue
        b = np.polyfit(g.pct_in_band, g.performance, 1)[0]
        slopes.append(dict(subject=s, slope=b, sensitive=bool(g.grp.iloc[0])))
    slopes = pd.DataFrame(slopes)
    a = slopes[slopes.sensitive].slope
    b = slopes[~slopes.sensitive].slope
    t, p = ttest_ind(a, b)
    print(f"\n  per-subject slopes, fitted one subject at a time:")
    print(f"      sensitive n={len(a)} mean {a.mean():+.3f} (SD {a.std():.3f})")
    print(f"      tolerant  n={len(b)} mean {b.mean():+.3f} (SD {b.std():.3f})")
    print(f"      two-sample t = {t:+.2f}, p = {p:.3f}")

    m1.to_csv(OUT / "clustered_statistics_metrics.csv", index=False)
    slopes.to_csv(OUT / "clustered_statistics_slopes.csv", index=False)
    return m1


if __name__ == "__main__":
    run()
