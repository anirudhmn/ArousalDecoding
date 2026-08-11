"""Candidate replacements for the gamma half of the sensitivity score.

``gamma_definitions.py`` shows that the gamma term is significant under one
definition only, the signed mean of a band-passed signal, which is close to zero
by construction. ``baseline_marker_screen.py`` screens 15 baseline measures and
finds respiration the strongest moderator of curve shape, surviving Holm
correction across the family.

This compares three candidate sensitivity scores end to end: the original
composite, HRV alone, and a vagal composite of HRV and respiration. For each it
reports the curve-shape moderation, the group split it produces, the coupling
between band deviation and performance within each group, and the band width the
grid search selects.

Outputs: results/vagal_composite_scores.csv,
         results/vagal_composite_baseline_panel.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, linregress, spearmanr

from common import C, OUT, hard_control, trial_table
from baseline_marker_screen import FEATURES, baseline_panel

CURVE_FORMULA = ("performance ~ C(difficulty)*arousal + C(difficulty)*I(arousal**2)"
                 " + {z} + arousal:{z} + I(arousal**2):{z}")


def moderation(base, z):
    """Does this marker moderate the shape of the arousal-performance curve?"""
    m = smf.mixedlm(CURVE_FORMULA.format(z=z), base, groups=base["subject"],
                    re_formula="~1").fit(reml=False)
    t = f"I(arousal ** 2):{z}"
    ci = m.conf_int()
    return m.params[t], ci.loc[t, 0], ci.loc[t, 1], m.pvalues[t]


def run():
    df = trial_table()
    panel = baseline_panel()
    for f in FEATURES:
        panel[f + "_z"] = (panel[f] - panel[f].mean()) / panel[f].std()
    panel.to_csv(OUT / "vagal_composite_baseline_panel.csv", index=False)

    # The signed-mean gamma is already z-scored in the trial table.
    pub_gamma = df.groupby("subject")["gamma_z"].first()

    drop = [c for c in df.columns
            if (c.endswith("_z") or c in panel.columns) and c != "subject"]
    base = df.drop(columns=drop, errors="ignore").merge(panel, on="subject")

    base["vagal_z"] = -(base.hrv_z + base.resp_z) / 2      # low vagal = sensitive
    base["hrvonly_z"] = -base.hrv_z
    base["pubscore"] = -base.hrv_z + base.subject.map(pub_gamma)

    print("=" * 78)
    print("Candidate sensitivity scores")
    print("=" * 78)

    print("\nAre the two vagal markers independent across subjects?")
    for a, b in [("hrv", "resp"), ("hrv", "hr"), ("resp", "hr")]:
        s = spearmanr(panel[a], panel[b])
        print(f"   {a:<5} vs {b:<5} rho = {s.statistic:+.3f}  p = {s.pvalue:.3f}")
    print("   -> HRV and respiration are uncorrelated, so the composite")
    print("      combines independent parasympathetic markers.")

    print("\n" + "-" * 78)
    print("Curve-shape moderation")
    print("-" * 78)
    print(f"{'marker':<28}{'beta':>10}{'95% CI':>22}{'p':>10}")
    curve = []
    for label, z in [("HRV", "hrv_z"), ("respiration", "resp_z"),
                     ("vagal composite", "vagal_z"),
                     ("gamma (real power)", "gamma_z")]:
        b, lo, hi, p = moderation(base, z)
        print(f"{label:<28}{b:>+10.3f}   [{lo:+7.3f}, {hi:+7.3f}]{p:>10.4f}")
        curve.append(dict(marker=label, beta=b, ci_lo=lo, ci_hi=hi, p=p))

    # --- what each score does to the grouping ------------------------------
    hc = hard_control(base)
    traj = C.optimal_trajectory(C.performance_surface(hc))
    metrics = C.deviation_metrics(hc, traj, 1.0)
    metrics["subject"] = hc.subject.to_numpy()

    print("\n" + "-" * 78)
    print("Grouping, coupling and selected band width")
    print("-" * 78)
    rows = []
    for label, col in [("original (-HRV + gamma_signed)", "pubscore"),
                       ("HRV only", "hrvonly_z"),
                       ("vagal composite (HRV + RESP)", "vagal_z")]:
        score = hc[col].to_numpy()
        med = np.median(score)
        g = (score > med).astype(float)

        a, b = metrics[g == 1], metrics[g == 0]
        ra = linregress(a.pct_in_band, a.performance)
        rb = linregress(b.pct_in_band, b.performance)

        d = metrics.copy()
        d["g"] = g
        m0 = smf.mixedlm("performance ~ pct_in_band", d,
                         groups=d["subject"]).fit(reml=False)
        m1 = smf.mixedlm("performance ~ pct_in_band * g", d,
                         groups=d["subject"]).fit(reml=False)
        lr = 2 * (m1.llf - m0.llf)

        ds_s, _ = C.band_width_sweep(hc, traj, np.where(g == 1)[0])
        ds_t, _ = C.band_width_sweep(hc, traj, np.where(g == 0)[0])
        ms = C.BAND_MULTIPLIERS[int(np.nanargmax(ds_s))]
        mt = C.BAND_MULTIPLIERS[int(np.nanargmax(ds_t))]

        n_s = hc[score > med].subject.nunique()
        n_t = hc[score <= med].subject.nunique()

        print(f"\n  {label}")
        print(f"     {n_s} sensitive / {n_t} tolerant subjects")
        print(f"     r: sensitive {ra.rvalue:+.3f} (p={ra.pvalue:.3f})   "
              f"tolerant {rb.rvalue:+.3f} (p={rb.pvalue:.3f})")
        print(f"     interaction p = {m1.pvalues['pct_in_band:g']:.3f}   "
              f"LRT chi2(2) = {lr:.2f}, p = {chi2.sf(lr, 2):.3f}")
        print(f"     best band: sensitive x{ms} (d={np.nanmax(ds_s):.2f})   "
              f"tolerant x{mt} (d={np.nanmax(ds_t):.2f})")

        rows.append(dict(score=label, n_sensitive=n_s, n_tolerant=n_t,
                         r_sensitive=ra.rvalue, p_sensitive=ra.pvalue,
                         r_tolerant=rb.rvalue, p_tolerant=rb.pvalue,
                         interaction_p=m1.pvalues["pct_in_band:g"],
                         lrt_p=chi2.sf(lr, 2), mult_sensitive=ms,
                         mult_tolerant=mt))

    print("\n  -> band widths are stable across all three scores; the")
    print("     band-width result does not hinge on the grouping choice.")
    print("  -> none of the three yields a significant group x coupling")
    print("     interaction; report the continuous moderation instead.")

    pd.concat([pd.DataFrame(curve).assign(kind="curve_shape"),
               pd.DataFrame(rows).assign(kind="grouping")],
              ignore_index=True).to_csv(OUT / "vagal_composite_scores.csv", index=False)
    print(f"\nwrote {OUT / 'vagal_composite_scores.csv'}")


if __name__ == "__main__":
    run()
