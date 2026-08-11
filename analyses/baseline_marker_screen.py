"""Screen baseline measures as markers of arousal sensitivity.

The composite ``-HRV_z + gamma_z`` depends on a gamma term that does not survive
a genuine power definition; see ``gamma_definitions.py``. This screens every
available baseline measure against three targets, with Holm correction across
the family:

  (a) does it moderate the shape of the arousal-performance curve?
  (b) does it moderate the coupling between band deviation and performance?
  (c) does it predict the per-subject best band width?

Target (c) is the practically useful one. If a baseline measure predicts how
wide a subject's control band should be, personalisation has a principled basis.

Outputs: results/baseline_marker_screen.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.signal import welch
from scipy.stats import spearmanr

from common import C, D, OUT, Y, cfg, hard_control, trial_table
from arousal.config import CH
from arousal.plotting import holm
from arousal.signals import extract_bandpowers_timeseries

BANDS = {"theta": slice(0, 64), "alpha": slice(64, 128),
         "beta": slice(128, 192), "gamma": slice(192, 256)}


def baseline_panel():
    """Per-subject baseline features from the calibration block.

    EEG bands are real power (mean squared band-passed amplitude), not the
    signed mean stored in the trial table.
    """
    ro = D.load_ring_epochs_online()
    calib = ro[ro.condition == cfg.CALIBRATION_CONDITION]
    rows = []
    for s, sid in enumerate(sorted(ro.subj_idx.unique())):
        g = calib[calib.subj_idx == sid]
        banded = np.concatenate(
            [extract_bandpowers_timeseries(x[None, ...])[0] for x in g["data"]],
            axis=1)
        r = {"subject": s}
        for name, sl in BANDS.items():
            r[name] = (banded[sl] ** 2).mean()
        r["theta_beta"] = r["theta"] / r["beta"]
        r["alpha_theta"] = r["alpha"] / r["theta"]
        for name, ch in [("hr", "HR"), ("hrv", "HRV-pNN35"), ("resp", "RESP"),
                         ("eda_p", "EDA-phasic"), ("eda_t", "EDA-tonic")]:
            r[name] = banded[256 + CH[ch] - 64].mean()
        r["pup"] = banded[[256 + CH["PUP-L"] - 64,
                           256 + CH["PUP-R"] - 64]].mean()
        # Variability, not just level - a plausible flexibility marker.
        r["hrv_sd"] = banded[256 + CH["HRV-pNN35"] - 64].std()
        r["hr_sd"] = banded[256 + CH["HR"] - 64].std()
        r["pup_sd"] = banded[[256 + CH["PUP-L"] - 64,
                              256 + CH["PUP-R"] - 64]].std()
        rows.append(r)
    return pd.DataFrame(rows)


FEATURES = ["hrv", "hr", "resp", "eda_p", "eda_t", "pup", "theta", "alpha",
            "beta", "gamma", "theta_beta", "alpha_theta", "hrv_sd", "hr_sd",
            "pup_sd"]


def run():
    df = trial_table()
    panel = baseline_panel()
    for f in FEATURES:
        panel[f + "_z"] = (panel[f] - panel[f].mean()) / panel[f].std()

    drop = [c for c in df.columns
            if (c.endswith("_z") or c in panel.columns) and c != "subject"]
    base = df.drop(columns=drop, errors="ignore").merge(panel, on="subject")
    hc = hard_control(base)
    traj = C.optimal_trajectory(C.performance_surface(hc))
    metrics = C.deviation_metrics(hc, traj, 1.0)
    metrics["subject"] = hc.subject.to_numpy()

    # Per-subject best band width, chosen on that subject's own trials.
    best = {}
    for s, g in hc.groupby("subject"):
        idx = np.where(hc.subject == s)[0]
        ds, _ = C.band_width_sweep(hc, traj, idx)
        if not np.all(np.isnan(ds)):
            best[s] = C.BAND_MULTIPLIERS[int(np.nanargmax(ds))]
    bw = pd.Series(best, name="best_mult")

    print("=" * 86)
    print("Screening baseline measures as arousal-sensitivity markers")
    print("=" * 86)
    print(f"\n{'feature':<13}{'(a) curve shape':>20}{'(b) coupling':>18}"
          f"{'(c) band width':>20}")
    print(f"{'':<13}{'beta':>10}{'p':>10}{'beta':>10}{'p':>8}{'rho':>12}{'p':>8}")

    res = []
    for f in FEATURES:
        z = f + "_z"
        ma = Y.fit_interaction_model(base, f)
        term = f"I(arousal ** 2):{z}"
        pa, ba = ma.pvalues[term], ma.params[term]

        m2 = metrics.copy()
        m2[z] = hc[z].to_numpy()
        mb = smf.mixedlm(f"performance ~ pct_in_band * {z}", m2,
                         groups=m2["subject"]).fit(reml=False)
        tb = f"pct_in_band:{z}"
        pb, bb = mb.pvalues[tb], mb.params[tb]

        sub = panel.set_index("subject")[z].reindex(bw.index)
        rho, pc = spearmanr(sub, bw)

        res.append(dict(f=f, ba=ba, pa=pa, bb=bb, pb=pb, rho=rho, pc=pc))
        print(f"{f:<13}{ba:>+10.3f}{pa:>10.4f}{bb:>+10.4f}{pb:>8.3f}"
              f"{rho:>+12.3f}{pc:>8.3f}")

    r = pd.DataFrame(res)
    print("\nHolm-corrected across the 15 features:")
    for col, label in [("pa", "(a) curve shape"), ("pb", "(b) coupling"),
                       ("pc", "(c) band width")]:
        adj = holm(r[col].to_numpy())
        surv = r.f[adj < 0.05].tolist()
        best_i = int(np.argmin(adj))
        print(f"   {label:<18} survivors: {surv if surv else 'none'}"
              f"   (best: {r.f[best_i]} at p_adj={adj[best_i]:.3f})")

    print(f"\nper-subject best multiplier: {dict(bw)}")
    print(f"   spread: {bw.min()} to {bw.max()}, median {bw.median()}")
    r.to_csv(OUT / "baseline_marker_screen.csv", index=False)


if __name__ == "__main__":
    run()
