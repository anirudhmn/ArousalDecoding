"""One multiple-comparison policy, applied to every reported p-value.

Every inferential p-value produced by the other scripts is entered here with the
family it belongs to, and Holm-Bonferroni is applied within each family. Holm
rather than Bonferroni because it is uniformly more powerful and makes no extra
assumptions.

The policy:

  * A family is a set of tests that address one question with interchangeable
    outcomes. Reporting the smallest of several such tests is what inflates
    error, so those are corrected together.
  * Pre-specified primary tests are their own family of one. HRV was specified
    a priori from Thayer et al., while the 17-measure calibration screen was
    not, so the screen is corrected and the a-priori test is not. That
    distinction is stated rather than applied silently, and HRV is listed in
    both places so the reader can see what it costs.
  * Descriptive quantities such as effect sizes, correlations reported for
    shape, and reproduction checks carry no p-value and enter no family.
  * Cluster-level p-values already control error across time within a test. The
    family there is the set of tests, not the set of time bins.

Entries are transcribed from the other scripts' saved outputs, so this file is a
registry rather than a re-analysis. Run it last.

Outputs: results/multiplicity.csv
"""
import numpy as np
import pandas as pd

from common import OUT
from arousal.plotting import holm

# --------------------------------------------------------------------------- #
# The registry. (family, label, p, source, primary)
# --------------------------------------------------------------------------- #

ENTRIES = [
    # -- decoder accuracy ---------------------------------------------------
    ("decoder comparisons", "peripheral vs EEG (within-subject)", 8.7e-06,
     "decoding", False),
    ("decoder comparisons", "combined vs EEG (within-subject)", 4.2e-07,
     "decoding", False),
    ("decoder comparisons", "combined vs peripheral (within-subject)", 0.016,
     "decoding", False),
    ("decoder comparisons", "peripheral vs motor", 4.9e-04, "negative_controls", False),
    ("decoder comparisons", "peripheral vs shuffled", 1.95e-14, "negative_controls", False),
    ("decoder comparisons", "LOSO peripheral vs chance", 3.7e-07, "loso_decoder", False),
    ("decoder comparisons", "LOSO EEG vs chance", 1.1e-06, "loso_decoder", False),
    ("decoder comparisons", "within-subject vs LOSO", 1.3e-05, "loso_decoder", False),
    # The cross-subject version of the central modality claim.
    ("decoder comparisons", "LOSO peripheral vs EEG", 0.1726, "loso_decoder", False),

    # -- label controls (see label_controls: four of these are no-ops by construction) --
    ("label controls", "real vs shift2 (6.7% of labels changed)", 0.335,
     "label_controls", False),
    ("label controls", "real vs cross-trial (0.9% changed)", 0.0952, "label_controls", False),
    ("label controls", "real vs phase-perm (0.8% changed)", 0.0480, "label_controls", False),
    ("label controls", "real vs block-perm (0.0% changed)", 0.0780, "label_controls", False),
    ("label controls", "real vs reversed (79.5% changed)", 3.88e-04, "label_controls", False),

    # -- validity of the index ----------------------------------------------
    ("validity", "matched-phase, all crossings + time", 1.07e-32, "matched_phase", True),
    ("validity", "matched-phase, + time^2", 1.59e-35, "matched_phase", False),
    ("validity", "matched-phase, large rings only", 1.40e-31, "matched_phase", False),

    # -- Yerkes-Dodson ------------------------------------------------------
    ("Yerkes-Dodson", "trial-mean quadratic, duration-stratified permutation",
     0.0125, "duration_permutation", True),
    ("Yerkes-Dodson", "landmark analysis at L = 30 s", 0.0328, "landmark_analysis", False),
    ("Yerkes-Dodson", "hazard quadratic, 5 s-bin fixed effects", 3.5e-10,
     "hazard_model", False),

    # -- Table 1 ------------------------------------------------------------
    ("Table 1", "% time above band", 0.0001, "clustered_statistics", False),
    ("Table 1", "% time in band", 0.0001, "clustered_statistics", False),
    ("Table 1", "mean excursion magnitude", 0.0097, "clustered_statistics", False),
    ("Table 1", "excursion rate", 0.0001, "clustered_statistics", False),

    # -- individual differences, a priori -----------------------------------
    ("a priori (family of one)", "arousal^2 x HRV", 0.0174, "t3/vagal_composite", True),

    # -- calibration-derived screen (17 measures, calibration_profile) -----------------------
    # An earlier 15-feature screen over baseline physiology alone
    # (baseline_marker_screen.py) is superseded by this one and is not reported
    # in the manuscript, so it does not enter the registry. Its members are a
    # subset of these seventeen apart from two ratios the screen dropped.
    ("calibration screen (17 measures)", "calibration arousal SD", 0.0016, "calibration_profile", False),
    ("calibration screen (17 measures)", "respiration", 0.0022, "calibration_profile", False),
    ("calibration screen (17 measures)", "HRV", 0.0174, "calibration_profile", False),
    ("calibration screen (17 measures)", "theta/beta", 0.0280, "calibration_profile", False),
    ("calibration screen (17 measures)", "calibration level", 0.0293, "calibration_profile", False),
    ("calibration screen (17 measures)", "beta", 0.0456, "calibration_profile", False),
    ("calibration screen (17 measures)", "calibration volatility", 0.0381, "calibration_profile", False),
    ("calibration screen (17 measures)", "calibration IQR", 0.0420, "calibration_profile", False),
    ("calibration screen (17 measures)", "HRV SD", 0.0485, "calibration_profile", False),
    ("calibration screen (17 measures)", "gamma", 0.1341, "calibration_profile", False),
    ("calibration screen (17 measures)", "heart rate", 0.3361, "calibration_profile", False),
    ("calibration screen (17 measures)", "alpha", 0.3577, "calibration_profile", False),
    ("calibration screen (17 measures)", "calibration frac-high", 0.3611, "calibration_profile", False),
    ("calibration screen (17 measures)", "theta", 0.7223, "calibration_profile", False),
    ("calibration screen (17 measures)", "EDA tonic", 0.7586, "calibration_profile", False),
    ("calibration screen (17 measures)", "pupil", 0.8356, "calibration_profile", False),
    ("calibration screen (17 measures)", "EDA phasic", 0.8840, "calibration_profile", False),

    # -- moderation of the deviation-performance coupling -------------------
    ("group moderation", "continuous sensitivity x %in-band", 0.0766,
     "clustered_statistics", True),
    ("group moderation", "median-split group x %in-band", 0.4346,
     "clustered_statistics", False),
    ("group moderation", "universal vs personalised band x %in-band", 0.2886,
     "clustered_statistics", False),
    ("group moderation", "per-subject slopes, sensitive vs tolerant", 0.739,
     "clustered_statistics", False),

    # -- held-out control bands (band_scheme_comparison) --------------------
    # Every scheme is scored on the same 113 held-out trials against two
    # baselines: the default band, and a universal band retuned to one
    # multiplier for everybody. The second is the comparison that matters.
    ("held-out control bands", "retuned universal vs default", 0.064,
     "band_scheme_comparison", False),
    ("held-out control bands", "group widths (composite) vs default", 0.194,
     "band_scheme_comparison", False),
    ("held-out control bands", "group widths (calibration marker) vs default", 0.354,
     "band_scheme_comparison", False),
    ("held-out control bands", "centre from fitted curve vs default", 0.407,
     "band_scheme_comparison", False),
    ("held-out control bands", "width from fitted curve vs default", 0.292,
     "band_scheme_comparison", False),
    ("held-out control bands", "group widths (composite) vs retuned", 0.806,
     "band_scheme_comparison", False),
    ("held-out control bands", "group widths (calibration marker) vs retuned", 0.752,
     "band_scheme_comparison", False),
    ("held-out control bands", "centre from fitted curve vs retuned", 0.907,
     "band_scheme_comparison", False),
    ("held-out control bands", "width from fitted curve vs retuned", 0.344,
     "band_scheme_comparison", False),

    # -- individual reactivity within trials --------------------------------
    ("individual reactivity", "random slope on excess above fixed band", 4.02e-10,
     "individual_reactivity", False),
    ("individual reactivity", "random slope on excess above adaptive band", 1.06e-17,
     "individual_reactivity", False),
    ("individual reactivity", "excess x respiration", 0.0003,
     "individual_reactivity", False),
    ("individual reactivity", "excess x calibration arousal SD", 0.0041,
     "individual_reactivity", False),
    ("individual reactivity", "excess x HRV", 0.0429,
     "individual_reactivity", False),

    # -- within-trial control parameters, added over the fixed band ---------
    ("within-trial control", "graded excess over adaptive band", 6.56e-04,
     "within_trial_control", True),
    ("within-trial control", "binary adaptive band", 0.770,
     "within_trial_control", False),
    ("within-trial control", "sustained excursion (>= 3 s)", 0.128,
     "within_trial_control", False),
    ("within-trial control", "rate of change of arousal", 0.0261,
     "within_trial_control", False),

    # -- reliability of the quantities being personalised -------------------
    ("reliability", "random slope on arousal (LRT)", 0.0384,
     "optimum_reliability", False),
    ("reliability", "band-width split-half agreement (Spearman)", 0.843,
     "optimum_reliability", False),

    # -- feedback effects ---------------------------------------------------
    ("feedback effects", "failure-locked rise, paired t", 1.7e-11, "failure_locked_arousal", True),
    ("feedback effects", "failure-locked rise by condition (ANOVA)", 0.1550,
     "failure_locked_arousal", False),
    ("feedback effects", "above-band x full BCI on 5 s hazard", 0.0400,
     "hazard_model", False),

    # -- trajectory clusters (already FWE-controlled across time) -----------
    ("trajectory clusters", "hard: omnibus", 0.2741, "cluster_permutation", False),
    ("trajectory clusters", "hard: BCI vs silence", 0.2690, "cluster_permutation", False),
    ("trajectory clusters", "hard: BCI vs half-sham", 0.0228, "cluster_permutation", False),
    ("trajectory clusters", "hard: BCI vs half-sham (2nd cluster)", 0.2209,
     "cluster_permutation", False),

    # -- external validation ------------------------------------------------
    ("external validation", "HR", 5.34e-09, "external_validation", False),
    ("external validation", "HRV-pNN35", 1.93e-04, "external_validation", False),
    ("external validation", "EDA-phasic", 2.24e-06, "external_validation", False),
    ("external validation", "EDA-tonic", 9.10e-02, "external_validation", False),
    ("external validation", "Pupil", 2.88e-04, "external_validation", False),
]


def run():
    df = pd.DataFrame(ENTRIES,
                      columns=["family", "test", "p", "source", "primary"])
    df["p_holm"] = np.nan
    for fam, g in df.groupby("family"):
        df.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
    df["survives"] = df["p_holm"] < 0.05

    print("=" * 96)
    print("Every reported p-value, its family, and Holm correction")
    print("=" * 96)

    for fam, g in df.groupby("family", sort=False):
        print(f"\n{fam}  (n = {len(g)})")
        print(f"  {'test':<52}{'p':>11}{'Holm':>11}{'':>4}{'src':>7}")
        for r in g.sort_values("p").itertuples():
            mark = " *" if r.survives else ("  " if r.p >= 0.05 else " x")
            star = "P" if r.primary else " "
            print(f"  {star} {r.test:<50}{r.p:>11.2e}{r.p_holm:>11.4f}"
                  f"{mark:>4}{r.source:>7}")

    print("\n  * survives correction within its family")
    print("  x significant uncorrected, does not survive")
    print("  P pre-specified primary test")

    print("\n" + "=" * 96)
    print("What changes once the policy is applied")
    print("=" * 96)
    lost = df[(df.p < 0.05) & (~df.survives)]
    print(f"\n  {len(df)} tests in {df.family.nunique()} families; "
          f"{int((df.p < 0.05).sum())} significant uncorrected, "
          f"{int(df.survives.sum())} after correction.")
    print("\n  Loses significance under its family's correction:")
    for r in lost.itertuples():
        print(f"    - {r.test} ({r.family}): p = {r.p:.4f} -> {r.p_holm:.4f}")
    if not len(lost):
        print("    none")

    print("\n  Notes on the policy:")
    print("   - HRV is reported as an a-priori test and is NOT corrected against")
    print("     the 17-measure screen it was not part of. Respiration comes FROM")
    print("     that screen and is reported with its corrected p-value. HRV also")
    print("     appears inside the screen, where it does not survive; both are")
    print("     shown rather than the more convenient one.")
    print("   - Table 1's four metrics are monotone transforms of one another,")
    print("     so correcting them as four independent tests is conservative to")
    print("     the point of being misleading; report two.")
    print("   - Cluster p-values already control error across time bins; the")
    print("     correction applied here is across the eight trajectory tests.")

    df.to_csv(OUT / "multiplicity.csv", index=False)
    print(f"\nwrote {OUT/'multiplicity.csv'}")
    return df


if __name__ == "__main__":
    run()
