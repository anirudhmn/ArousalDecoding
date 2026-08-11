"""Is the peripheral-over-EEG result an artefact of scaling?

The comparison between the peripheral index and the EEG feedback signal is only
meaningful if the two are scaled the same way. They are not.

  peripheral   ``continuous_arousal`` rescales the decoder probability against
               the 5th and 95th percentiles of that subject's training
               distribution and clips to 0-100. Between-trial differences in
               level survive.
  EEG          ``data._raw_metric_traces`` min-max scales every channel within
               each trial. Every trial then spans exactly 0-100, so the trial
               mean reflects trace shape rather than arousal level.

A signal that is min-max scaled per trial has most of its between-trial variance
in the mean removed by construction, so it cannot show a relationship between
arousal level and performance whatever the EEG decoder knows. The EEG null could
therefore be a preprocessing artefact rather than a decoder result.

Two directions are tested, because either one alone would be deniable:

  (a) rescale the EEG signal per subject, the way the peripheral index is
      scaled, and refit the quadratic model;
  (b) rescale the peripheral index per trial, the way the EEG signal is scaled,
      and refit. If the peripheral inverted-U also disappears, scaling is
      demonstrably the operative difference.

Matched-phase course discrimination is then run on every variant.

Outputs: results/scaling_parity_yerkes.csv,
         results/scaling_parity_matched_phase.csv,
         results/scaling_parity_traces.pkl
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import D, OUT, Y, cfg, trial_table
from common import FS
from arousal.config import CH
from arousal.signals import asymmetric_iir

FB_CHANNEL = CH["FB-HB-0-1-nrm"]      # the signal that drove the auditory loop
TRACES = OUT / "scaling_parity_traces.pkl"


# --------------------------------------------------------------------------- #
# Rescaling variants
# --------------------------------------------------------------------------- #

def minmax_per_trial(x):
    """What `_raw_metric_traces` does: 0-100 within each trial."""
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo + 1e-8) * 100


def percentile_scale(x, lower, upper):
    """What `continuous_arousal` does: fixed per-subject bounds, then clip."""
    return np.clip((x - lower) / (upper - lower + 1e-12), 0, 1) * 100


def smooth_like_decoder(x, update=16):
    """Apply the decoder's warm-up and asymmetric IIR to an arbitrary trace.

    Only used for the full-parity variant: it makes the EEG signal go through
    every step the peripheral index goes through, so that any remaining
    difference is attributable to the decoder rather than the pipeline.
    """
    out = np.zeros_like(x, dtype=float)
    if len(x) <= 256:
        return out
    a_up, a_down = asymmetric_iir(update)
    S = 0.0
    for i in range(256, len(x), update):
        A = float(x[i])
        alpha = a_up if A > S else a_down
        S = (1 - alpha) * S + alpha * A
        out[i:min(i + update, len(x))] = S
    return out


# --------------------------------------------------------------------------- #
# Raw feedback traces, rebuilt without the per-trial scaling
# --------------------------------------------------------------------------- #

def raw_feedback_traces():
    """Unscaled FB-HB-0-1-nrm per (subj_id, trial_idx), plus calibration values.

    `build_trial_table` only ever stores the min-max version, so the native
    signal has to be reassembled from the epoch file.
    """
    ro = D.load_ring_epochs_online()
    closed = ro[ro.condition != cfg.CALIBRATION_CONDITION]
    calib = ro[ro.condition == cfg.CALIBRATION_CONDITION]

    traces, bounds = {}, {}
    for subj_id, sg in closed.groupby("subj_idx"):
        own = {}
        for trial_idx, g in sg.groupby("trial_idx", sort=True):
            g = g.sort_values("ring_idx")
            own[(subj_id, trial_idx)] = np.concatenate(
                [r[FB_CHANNEL] for r in g["data"]])
        traces.update(own)

        cl_vals = np.concatenate(list(own.values()))
        cl = (np.nanpercentile(cl_vals, 5), np.nanpercentile(cl_vals, 95))

        # The calibration block is open-loop, so the feedback channel may carry
        # nothing there. Fall back to the closed-loop distribution when it does
        # not, and say so rather than silently producing NaN bounds.
        cal = calib[calib.subj_idx == subj_id]
        cal_vals = (np.concatenate([r[FB_CHANNEL] for r in cal["data"]])
                    if len(cal) else np.array([np.nan]))
        finite = np.isfinite(cal_vals)
        if finite.sum() > 100 and np.nanstd(cal_vals) > 1e-12:
            cal_b = (np.nanpercentile(cal_vals, 5), np.nanpercentile(cal_vals, 95))
        else:
            cal_b = cl
            bounds.setdefault("_fallback", []).append(subj_id)

        bounds[subj_id] = {"cal": cal_b, "cl": cl}

    fb = bounds.pop("_fallback", [])
    if fb:
        print(f"[note] calibration-block feedback channel unusable for "
              f"{len(fb)} subject(s); using closed-loop percentiles instead")
    return traces, bounds


def build_variants(df):
    """Attach every scaling variant to the trial table as a new trace column.

    Only the native traces and the per-subject bounds are cached; every scaled
    variant is rederived, so changing a scaling rule does not require another
    pass over the 4 GB epoch file.
    """
    if TRACES.exists():
        cache = pd.read_pickle(TRACES)
        traces, bounds = cache["traces"], cache["bounds"]
    else:
        traces, bounds = raw_feedback_traces()
        pd.to_pickle({"traces": traces, "bounds": bounds}, TRACES)

    cols = {k: [] for k in ("eeg_raw", "eeg_trial", "eeg_subj_cal",
                            "eeg_subj_cl", "eeg_subj_full", "periph_trial")}
    for r in df.itertuples():
        n = len(r.new_arousal)
        raw = traces.get((r.subj_id, r.trial_idx))
        if raw is None:
            raw = np.full(n, np.nan)
        raw = raw[:n] if len(raw) >= n else np.pad(raw, (0, n - len(raw)), "edge")

        lo_cal, hi_cal = bounds[r.subj_id]["cal"]
        lo_cl, hi_cl = bounds[r.subj_id]["cl"]

        cols["eeg_raw"].append(raw)
        cols["eeg_trial"].append(minmax_per_trial(raw))
        cols["eeg_subj_cal"].append(percentile_scale(raw, lo_cal, hi_cal))
        cols["eeg_subj_cl"].append(percentile_scale(raw, lo_cl, hi_cl))
        cols["eeg_subj_full"].append(
            smooth_like_decoder(percentile_scale(raw, lo_cl, hi_cl)))
        cols["periph_trial"].append(minmax_per_trial(r.new_arousal))

    for k, v in cols.items():
        df[k] = v
    return df


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #

VARIANTS = [
    ("peripheral (per-subject)", "new_arousal", "per-subject percentile + clip + IIR"),
    ("peripheral, per-trial minmax", "periph_trial", "REVERSE TEST"),
    ("EEG (per-trial)", "old_arousal", "per-trial min-max"),
    ("EEG, per-trial minmax (rebuilt)", "eeg_trial", "reproduces the stored column"),
    ("EEG, per-subject percentile", "eeg_subj_cl", "matches peripheral scaling"),
    ("EEG, per-subject + IIR + warm-up", "eeg_subj_full", "full pipeline parity"),
    ("EEG, native units", "eeg_raw", "no scaling at all"),
]

# The feedback channel is all-NaN in the open-loop calibration block, so its
# per-subject bounds have to come from the closed-loop trials themselves. That
# is a mild in-sample advantage for the EEG variants relative to the peripheral
# index, whose bounds come from a held-out training distribution. The parity
# test therefore favours EEG if it is biased at all.


def yerkes_table(df):
    print("=" * 96)
    print("(a) Yerkes-Dodson fit under matched scaling")
    print("=" * 96)
    print(f"\n{'index':<34}{'scaling':<36}{'beta_quad':>11}{'p_easy':>10}"
          f"{'p_hard':>10}{'dAIC':>8}{'opt':>8}")

    rows = []
    for name, col, note in VARIANTS:
        d = df.copy()
        d["mean_val"] = [np.mean(x) for x in d[col]]
        d = d[np.isfinite(d["mean_val"])]
        if len(d) < 50 or not np.isfinite(d["mean_val"].std()) \
                or d["mean_val"].std() < 1e-9:
            print(f"{name:<34}{note:<36}   unusable "
                  f"(n={len(d)}, SD={d['mean_val'].std():.3g})")
            continue
        mq = Y.fit_yd_model(d, arousal_col="mean_val")
        ml = Y.fit_yd_model(d, arousal_col="mean_val",
                            formula=Y.FORMULA_LINEAR)
        p = Y.quadratic_pvalues(mq)
        opt, _ = Y.optimum_with_ci(mq, difficulty=1)
        b2 = mq.params[Y.QUAD]
        ci = mq.conf_int()
        flag = " *" if (b2 < 0 and p[0] < 0.05 and mq.aic - ml.aic < 0) else ""
        print(f"{name:<34}{note:<36}{b2:>+11.3f}{p[0]:>10.2e}{p[1]:>10.2e}"
              f"{mq.aic-ml.aic:>+8.1f}{opt:>8.1f}{flag}")
        rows.append(dict(index=name, column=col, scaling=note, n=len(d),
                         beta_quad=b2, ci_lo=ci.loc[Y.QUAD, 0],
                         ci_hi=ci.loc[Y.QUAD, 1], p_easy=p[0], p_hard=p[1],
                         daic=mq.aic - ml.aic, optimum_hard=opt,
                         sd_of_trial_means=d["mean_val"].std()))

    print("\n(* = negative quadratic, significant on the easy course, and the")
    print(" quadratic model beats the linear one)")
    return pd.DataFrame(rows)


def variance_check(df):
    """How much between-trial variance does each scaling leave in the mean?"""
    print("\n" + "-" * 96)
    print("Between-trial variance in the trial mean (the predictor in equation 1)")
    print("-" * 96)
    print(f"\n{'index':<34}{'SD of trial means':>20}{'SD within subject':>20}")
    for name, col, _ in VARIANTS:
        d = df.dropna(subset=[col]).copy()
        d["m"] = [np.mean(x) for x in d[col]]
        within = d.groupby("subject")["m"].std().mean()
        print(f"{name:<34}{d['m'].std():>20.2f}{within:>20.2f}")
    print("\nPer-trial min-max scaling removes level information by construction;")
    print("what is left is trace shape.")


def matched_phase(df):
    """Matched-phase discrimination, applied to every index including EEG."""
    print("\n" + "=" * 96)
    print("(b) matched-phase course discrimination, all indices")
    print("=" * 96)
    print("\nA pure time-ramp predicts beta_hard = 0. Hard rings are smaller than")
    print("easy rings at every elapsed time.")
    print(f"\n{'index':<34}{'model':<30}{'beta_hard':>11}{'p':>13}{'n':>8}")

    rows = []
    for name, col, _ in VARIANTS:
        if col == "eeg_raw":
            continue                       # native units are not comparable
        recs = []
        for r in df.itertuples():
            if not r.events:
                continue
            a = getattr(r, col)
            if a is None or np.all(np.isnan(a)):
                continue
            for size, samp in r.events:
                if samp >= len(a) or samp < FS:
                    continue
                recs.append({"subject": r.subject, "difficulty": r.difficulty,
                             "size": size, "t_s": samp / FS,
                             "arousal": np.mean(a[max(0, samp - FS):samp])})
        e = pd.DataFrame(recs).dropna()
        lg = e[e["size"] == "large"]
        for label, frame, formula in [
                ("all crossings, + time", e, "arousal ~ difficulty + t_s"),
                ("all crossings, + time + t^2", e,
                 "arousal ~ difficulty + t_s + I(t_s**2)"),
                ("large rings only, + time", lg, "arousal ~ difficulty + t_s")]:
            m = smf.mixedlm(formula, frame, groups=frame["subject"]).fit(reml=False)
            b, p = m.params["difficulty"], m.pvalues["difficulty"]
            print(f"{name:<34}{label:<30}{b:>+11.2f}{p:>13.2e}{len(frame):>8d}")
            rows.append(dict(index=name, column=col, model=label,
                             beta=b, p=p, n=len(frame)))
        print()
    return pd.DataFrame(rows)


def run():
    df = build_variants(trial_table())
    yd = yerkes_table(df)
    variance_check(df)
    mp = matched_phase(df)

    yd.to_csv(OUT / "scaling_parity_yerkes.csv", index=False)
    mp.to_csv(OUT / "scaling_parity_matched_phase.csv", index=False)
    print(f"\nwrote {OUT/'scaling_parity_yerkes.csv'} and {OUT/'scaling_parity_matched_phase.csv'}")
    return yd, mp


if __name__ == "__main__":
    run()
