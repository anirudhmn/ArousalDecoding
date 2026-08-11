"""Is the gamma by arousal-squared interaction robust to how gamma is defined?

The baseline gamma feature is the mean of a signed band-passed signal, which has
a near-zero mean by construction. This computes five definitions of the same
quantity and asks whether the interaction survives each of them.

Outputs: printed table only
"""
import numpy as np
import pandas as pd
from scipy.signal import hilbert, welch

from common import D, OUT, Y, cfg
from arousal.config import CH
from arousal.signals import extract_bandpowers_timeseries

GAMMA = slice(192, 256)   # the 64 gamma-band channels after band-passing


def gamma_variants():
    ro = D.load_ring_epochs_online()
    calib = ro[ro.condition == cfg.CALIBRATION_CONDITION]
    rows = []
    for s, subj_id in enumerate(sorted(ro.subj_idx.unique())):
        g = calib[calib.subj_idx == subj_id]
        banded = np.concatenate(
            [extract_bandpowers_timeseries(x[None, ...])[0] for x in g["data"]],
            axis=1)
        gam = banded[GAMMA]
        raw = np.concatenate([x[:64] for x in g["data"]], axis=1)
        f, psd = welch(raw, fs=cfg.FS, nperseg=512, axis=-1)
        band = (f >= 32) & (f <= 55)
        rows.append({
            "subject": s,
            "signed_mean": gam.mean(),                      # the stored baseline feature
            "power": (gam ** 2).mean(),                     # mean square
            "variance": gam.var(),                          # variance
            "envelope": np.abs(hilbert(gam, axis=-1)).mean(),
            "welch": psd[:, band].mean(),
            "log_power": np.log((gam ** 2).mean()),
            # peripheral features are unchanged across variants
            "hrv": banded[256 + CH["HRV-pNN35"] - 64].mean(),
        })
    return pd.DataFrame(rows)


def run():
    df = pd.read_pickle(
        __import__("common").ROOT / "data" / "trial_table.pkl")
    gv = gamma_variants()

    print("=" * 78)
    print("Gamma x arousal^2 interaction under five definitions of gamma")
    print("=" * 78)
    print("\nper-subject values:")
    print(gv.drop(columns=["hrv"]).round(6).to_string(index=False))

    variants = ["signed_mean", "power", "variance", "envelope", "welch", "log_power"]
    print("\nrank correlation with the signed_mean version:")
    for v in variants[1:]:
        print(f"   {v:<12} rho = {gv['signed_mean'].corr(gv[v], method='spearman'):+.3f}")

    print(f"\n{'gamma definition':<14}{'beta':>10}{'p':>10}   {'verdict':<12}"
          f"{'group flips':>12}")
    base_groups = None
    for v in variants:
        d = df.drop(columns=["gamma", "gamma_z"], errors="ignore").merge(
            gv[["subject", v]].rename(columns={v: "gamma"}), on="subject")
        d["gamma_z"] = (d["gamma"] - d["gamma"].mean()) / d["gamma"].std()
        m = Y.fit_interaction_model(d, "gamma")
        term = "I(arousal ** 2):gamma_z"
        beta, p = m.params[term], m.pvalues[term]

        per = d.groupby("subject")[["hrv_z", "gamma_z"]].first()
        score = -per["hrv_z"] + per["gamma_z"]
        sens = frozenset(score[score > score.median()].index)
        if base_groups is None:
            base_groups, flips = sens, 0
        else:
            flips = len(base_groups ^ sens)
        print(f"{v:<14}{beta:>+10.3f}{p:>10.4f}   "
              f"{'significant' if p < 0.05 else 'not significant':<12}{flips:>12}")

    print("\n(the stored feature is signed_mean; 'group flips' counts "
          "subjects\n changing sensitivity group relative to it)")

    # HRV, for contrast - unchanged by any of this.
    m = Y.fit_interaction_model(df, "hrv")
    t = "I(arousal ** 2):hrv_z"
    print(f"\nfor contrast, arousal^2 x HRV: beta={m.params[t]:+.3f}, "
          f"p={m.pvalues[t]:.4f}  (unaffected - HRV has one definition)")


if __name__ == "__main__":
    run()
