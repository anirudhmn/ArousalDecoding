"""Do subjects differ in how strongly excess arousal costs them, and can a band
derived from that difference track performance better than a universal one?

``band_width_limits.py`` shows that band width fails as a control parameter
because it is a per-subject constant and the arousal-performance association
lives within subjects. That argument applies to any per-subject constant, so it
raises an obvious question: is there a per-subject quantity worth setting at
all, and if so which one?

Three parts, each prior to the next.

  A  Is reactivity to excess arousal subject-specific? A random slope on the
     graded excess above the band, tested against a random intercept only. If
     subjects do not differ here, nothing downstream can help.

  B  Does a calibration measure predict that reactivity? An interaction between
     the baseline marker and the graded excess in the same hazard model. This is
     the within-trial counterpart of the curve-shape moderation reported by
     ``calibration_profile.py``, and it uses a different unit of analysis, so
     agreement between them is not automatic.

  C  Does a band derived from the subject's own fitted curve beat a universal
     band? Leave-one-subject-out throughout: the optimal trajectory and the
     curve-shape interaction are both fitted on the training subjects, and the
     held-out subject's band comes from their calibration marker through that
     model. Two parameters are derived separately, because they are not
     interchangeable.

       width   If performance falls as c(a - a*)^2 then a fixed tolerated loss
               implies a half-width proportional to 1/sqrt(|c|). A subject with
               a steeper curve gets a tighter band.
       centre  The subject's own fitted optimum replaces the group optimum, so
               the whole trajectory shifts rather than widening.

     A universal band at the best single multiplier is included as the control
     that matters: a personalised scheme has to beat a retuned global one, not
     merely the default.

Outputs: results/individual_reactivity_slopes.csv,
         results/individual_reactivity_markers.csv,
         results/individual_reactivity_folds.csv,
         results/individual_reactivity_bands.csv
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, pearsonr, spearmanr

# ``C`` is imported under another name: patsy resolves the categorical
# marker C() from the caller's namespace, and statsmodels' formula API
# for OLS would otherwise find the control module instead.
from common import C as ctrl
from common import OUT, hard_control, trial_table

warnings.filterwarnings("ignore")

MARKER = "cal_sd"                  # the measure that survives the calibration screen
MARKERS = ["cal_sd", "resp", "hrv"]
GRID_LO, GRID_HI = 0.5, 3.0        # the band-width grid used elsewhere
MAX_SHIFT = 20.0                   # cap on the centre shift, in arousal units
RNG = np.random.default_rng(11)
N_BOOT = 10000

FORMULA = ("performance ~ C(difficulty)*arousal + C(difficulty)*I(arousal**2)"
           " + mz + arousal:mz + I(arousal**2):mz")


def _profile():
    prof = pd.read_csv(OUT / "calibration_profile.csv")
    for m in MARKERS:
        prof[m + "_z"] = (prof[m] - prof[m].mean()) / prof[m].std()
    return prof


def _lrt(m0, m1):
    d = 2 * (m1.llf - m0.llf)
    k = len(m1.params) - len(m0.params)
    return d, k, chi2.sf(d, k)


# --------------------------------------------------------------------------- #
# A: is reactivity subject-specific?
# --------------------------------------------------------------------------- #

def part_a(bins):
    print("=" * 78)
    print("(A) is reactivity to excess arousal subject-specific?")
    print("=" * 78)
    rows = []
    for col in ("excess_fixed", "excess_adaptive"):
        m0 = smf.mixedlm(f"crash ~ {col} + C(tb)", bins,
                         groups=bins.subject).fit(reml=False)
        m1 = smf.mixedlm(f"crash ~ {col} + C(tb)", bins, groups=bins.subject,
                         re_formula=f"~{col}").fit(reml=False)
        d, k, p = _lrt(m0, m1)
        print(f"  {col:<17} random slope LRT chi2({k}) = {d:6.2f}, p = {p:.2e}")
        rows.append(dict(signal=col, chi2=d, df=k, p=p))

    slopes = {}
    for s, g in bins.groupby("subject"):
        if g.crash.sum() < 5 or g.excess_adaptive.std() < 1e-9:
            continue
        try:
            slopes[s] = smf.ols("crash ~ excess_adaptive + C(tb)",
                                g).fit().params["excess_adaptive"]
        except Exception:
            pass
    sl = pd.Series(slopes, name="slope")
    print(f"\n  per-subject slope on excess_adaptive, fitted independently:")
    print(f"    n = {len(sl)}  mean {sl.mean():+.4f}  SD {sl.std():.4f}"
          f"  range {sl.min():+.4f} to {sl.max():+.4f}")
    print("  Subjects differ, so a per-subject quantity is at least well defined.")
    pd.DataFrame(rows).to_csv(OUT / "individual_reactivity_slopes.csv", index=False)
    return sl


# --------------------------------------------------------------------------- #
# B: does a baseline measure predict it?
# --------------------------------------------------------------------------- #

def part_b(bins, slopes, prof):
    print("\n" + "=" * 78)
    print("(B) does a calibration measure predict reactivity?")
    print("=" * 78)
    print(f"\n{'signal':<17}{'marker':<12}{'beta':>11}{'95% CI':>24}{'p':>10}")
    rows = []
    for col in ("excess_fixed", "excess_adaptive"):
        for m in MARKERS:
            z = m + "_z"
            fit = smf.mixedlm(f"crash ~ {col}*{z} + C(tb)", bins,
                              groups=bins.subject).fit(reml=False)
            term = f"{col}:{z}"
            ci = fit.conf_int()
            print(f"{col:<17}{m:<12}{fit.params[term]:>+11.5f}"
                  f"   [{ci.loc[term, 0]:+.5f}, {ci.loc[term, 1]:+.5f}]"
                  f"{fit.pvalues[term]:>10.4f}")
            rows.append(dict(signal=col, marker=m, beta=fit.params[term],
                             ci_lo=ci.loc[term, 0], ci_hi=ci.loc[term, 1],
                             p=fit.pvalues[term]))
    print("\n  A negative coefficient means a higher marker weakens the link")
    print("  between excess arousal and imminent failure, which is the same")
    print("  direction as the curve-shape moderation at trial level.")

    pm = prof.set_index("subject")
    print("\n  per-subject slope against each marker "
          f"(n = {len(slopes)} subjects, one point each):")
    for m in MARKERS:
        x = pm.loc[slopes.index, m + "_z"].to_numpy()
        r, p = pearsonr(x, slopes.to_numpy())
        rho, prho = spearmanr(x, slopes.to_numpy())
        print(f"    {m:<12} r = {r:+.3f} (p = {p:.3f})   "
              f"rho = {rho:+.3f} (p = {prho:.3f})")
    print("  These are null. The mixed model uses every bin and models the")
    print("  clustering; the 16-point correlation discards that and is")
    print("  dominated by one extreme subject. Report both.")
    pd.DataFrame(rows).to_csv(OUT / "individual_reactivity_markers.csv", index=False)


# --------------------------------------------------------------------------- #
# C: a band derived from the fitted curve
# --------------------------------------------------------------------------- #

def _curve(fit, m):
    """Linear and quadratic coefficients on the hard course at marker value m."""
    p = fit.params
    b1 = (p["arousal"] + p.get("C(difficulty)[T.1]:arousal", 0.0)
          + p["arousal:mz"] * m)
    b2 = (p["I(arousal ** 2)"] + p.get("C(difficulty)[T.1]:I(arousal ** 2)", 0.0)
          + p["I(arousal ** 2):mz"] * m)
    return b1, b2


def _shifted(traj, delta):
    out = dict(traj)
    out["optimal"] = traj["optimal"] + delta
    return out


def part_c(df, prof):
    print("\n" + "=" * 78)
    print("(C) a control band derived from the held-out subject's fitted curve")
    print("=" * 78)

    df = df.merge(prof[["subject", MARKER + "_z"]].rename(
        columns={MARKER + "_z": "mz"}), on="subject", how="left")
    hc = hard_control(df)
    subs = sorted(hc.subject.unique())
    print(f"\n{len(hc)} hard-course control trials, {len(subs)} subjects, "
          f"marker = {MARKER}")

    rows = {k: [] for k in ("universal", "retuned", "width", "centre", "both")}
    folds = []
    for s in subs:
        train = hc[hc.subject != s].reset_index(drop=True)
        test = hc[hc.subject == s].reset_index(drop=True)
        if not len(test):
            continue

        traj = ctrl.optimal_trajectory(ctrl.performance_surface(train))
        fit = smf.mixedlm(FORMULA, train, groups=train["subject"],
                          re_formula="~1").fit(reml=False)

        m_test = float(test.mz.iloc[0])
        m_train = train.groupby("subject").mz.first()
        curv_train = np.array([_curve(fit, m)[1] for m in m_train])
        curv_test = _curve(fit, m_test)[1]

        # A positive curvature means the fitted curve has no interior optimum,
        # so the width is undefined; floor it at the steepest sensible value
        # rather than letting it diverge.
        concave = np.abs(curv_train[curv_train < 0])
        floor = np.percentile(concave, 5) * 0.2 if concave.size else 1.0
        width = lambda c: 1.0 / np.sqrt(max(abs(min(c, -1e-6)), floor))
        scale = np.median([width(c) for c in curv_train])
        mult = float(np.clip(width(curv_test) / scale, GRID_LO, GRID_HI))

        b1, b2 = _curve(fit, m_test)
        opt_test = -b1 / (2 * b2) if b2 < 0 else np.nan
        opts = [(-_curve(fit, m)[0] / (2 * _curve(fit, m)[1]))
                if _curve(fit, m)[1] < 0 else np.nan for m in m_train]
        opt_train = np.nanmedian(opts)
        shift = (float(np.clip(opt_test - opt_train, -MAX_SHIFT, MAX_SHIFT))
                 if np.isfinite(opt_test) else 0.0)

        for key, tj, mm in (("universal", traj, 1.0),
                            ("retuned", traj, 1.75),
                            ("width", traj, mult),
                            ("centre", _shifted(traj, shift), 1.0),
                            ("both", _shifted(traj, shift), mult)):
            d = ctrl.deviation_metrics(test, tj, mm)
            d["subject"] = s
            rows[key].append(d)

        folds.append(dict(subject=s, marker_z=m_test, curvature=curv_test,
                          multiplier=mult, optimum_subject=opt_test,
                          optimum_group=opt_train, centre_shift=shift))

    folds = pd.DataFrame(folds)
    print("\nPer-fold band parameters, all derived on training subjects only:")
    print(folds.round(3).to_string(index=False))
    print(f"\n  width multiplier {folds.multiplier.min():.2f} to "
          f"{folds.multiplier.max():.2f} (median {folds.multiplier.median():.2f});"
          f" centre shift {folds.centre_shift.min():+.1f} to "
          f"{folds.centre_shift.max():+.1f}")

    frames = {k: pd.concat(v, ignore_index=True) for k, v in rows.items()}
    labels = {"universal": "universal x1.0",
              "retuned": "universal x1.75 (no personalisation)",
              "width": "width from curvature",
              "centre": "centre from fitted optimum",
              "both": "centre + width"}

    print("\nHeld-out trials pooled across folds:")
    print(f"{'band':<38}{'n':>5}{'r':>9}{'p':>10}{'Cohen d':>10}")
    summary = []
    for key, frame in frames.items():
        r, p = pearsonr(frame.pct_in_band, frame.performance)
        d, dp = ctrl.cohens_d_good_bad(frame)
        print(f"{labels[key]:<38}{len(frame):>5}{r:>+9.3f}{p:>10.4f}{d:>10.2f}")
        summary.append(dict(band=labels[key], n=len(frame), r=r, p=p,
                            cohens_d=d, d_p=dp))

    # Subject-level bootstrap of the difference in r, which respects clustering.
    base = frames["universal"]
    subj = base.subject.to_numpy()
    index = {s: np.where(subj == s)[0] for s in np.unique(subj)}
    uniq = np.unique(subj)

    def delta_r(a, b):
        out = np.empty(N_BOOT)
        ax, ay = a.pct_in_band.to_numpy(), a.performance.to_numpy()
        bx, by = b.pct_in_band.to_numpy(), b.performance.to_numpy()
        for i in range(N_BOOT):
            pick = RNG.choice(uniq, uniq.size, replace=True)
            k = np.concatenate([index[s] for s in pick])
            out[i] = (np.corrcoef(bx[k], by[k])[0, 1]
                      - np.corrcoef(ax[k], ay[k])[0, 1])
        lo, hi = np.percentile(out, [2.5, 97.5])
        return out.mean(), lo, hi, 2 * min((out <= 0).mean(), (out >= 0).mean())

    print(f"\n{'comparison':<46}{'delta r':>9}{'95% CI':>21}{'p':>8}")
    for name, a, b in (("centre vs universal x1.0", "universal", "centre"),
                       ("width vs universal x1.0", "universal", "width"),
                       ("retuned x1.75 vs universal x1.0", "universal", "retuned"),
                       ("centre vs retuned x1.75", "retuned", "centre")):
        m, lo, hi, p = delta_r(frames[a], frames[b])
        print(f"{name:<46}{m:>+9.3f}   [{lo:+.3f}, {hi:+.3f}]{p:>8.3f}")
        summary.append(dict(band=f"DELTA {name}", n=len(base), r=m,
                            p=p, cohens_d=np.nan, d_p=np.nan))

    print("\n  Reading. Width from curvature does nothing, which is the same")
    print("  conclusion band_width_limits.py reaches for an arbitrary mapping,")
    print("  now under one derived from the fitted curve. Moving the band")
    print("  centre is the only scheme that improves both metrics, but its")
    print("  interval spans zero and it does not beat a retuned universal band.")

    folds.to_csv(OUT / "individual_reactivity_folds.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "individual_reactivity_bands.csv",
                                 index=False)
    for key, frame in frames.items():
        frame.to_csv(OUT / f"individual_reactivity_band_{key}.csv", index=False)


def run():
    prof = _profile()
    bins = pd.read_csv(OUT / "within_trial_control_bins.csv")
    bins = bins.merge(prof[["subject"] + [m + "_z" for m in MARKERS]],
                      on="subject", how="left")
    bins = bins.dropna(subset=["excess_fixed", "excess_adaptive", "crash",
                               "cal_sd_z"])
    print(f"{len(bins)} one-second bins, {bins.subject.nunique()} subjects\n")

    slopes = part_a(bins)
    part_b(bins, slopes, prof)
    part_c(trial_table(), prof)


if __name__ == "__main__":
    run()
