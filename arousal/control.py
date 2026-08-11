"""Time-varying optimal arousal trajectory, deviation metrics and control bands."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_ind

from .config import FS

TIME_BIN_SEC = 1
AROUSAL_BIN = 5
SAMPLES_PER_BIN = FS * TIME_BIN_SEC

MIN_SAMPLES_PER_BIN = 5      # bins with fewer observations are ignored
MIN_CONFIDENCE_FOR_FIT = 20  # time bins entering the polynomial fit
POLY_DEGREE = 3
MIN_STD = 5.0                # floor on the band half-width

BAND_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]


def _binned_arousal(trace):
    """Mean decoded arousal in each complete 1 s bin of a trial."""
    n = len(trace) // SAMPLES_PER_BIN
    return np.array([trace[i * SAMPLES_PER_BIN:(i + 1) * SAMPLES_PER_BIN].mean()
                     for i in range(n)])


def performance_surface(trials_df):
    """Mean flight time as a joint function of decoded arousal and trial time.

    Returns the raw and smoothed surfaces plus the bin edges. The smoothed
    version linearly interpolates empty interior bins before Gaussian blurring;
    bins with no observations at all are restored to NaN.
    """
    durations = trials_df["performance"].to_numpy() / FS
    all_arousal = np.concatenate(trials_df["new_arousal"].to_numpy())
    lo, hi = np.percentile(all_arousal, [1, 99])

    arousal_bins = np.arange(np.floor(lo / AROUSAL_BIN) * AROUSAL_BIN,
                             np.ceil(hi / AROUSAL_BIN) * AROUSAL_BIN + AROUSAL_BIN,
                             AROUSAL_BIN)
    time_bins = np.arange(0, int(np.ceil(durations.max())) + TIME_BIN_SEC, TIME_BIN_SEC)
    n_a, n_t = len(arousal_bins) - 1, len(time_bins) - 1

    perf_sum = np.zeros((n_a, n_t))
    perf_count = np.zeros((n_a, n_t))
    for _, row in trials_df.iterrows():
        perf = row["performance"] / FS
        for t_idx, w_ar in enumerate(_binned_arousal(row["new_arousal"])):
            if t_idx >= n_t:
                break
            a_idx = np.clip(np.searchsorted(arousal_bins, w_ar, side="right") - 1,
                            0, n_a - 1)
            perf_sum[a_idx, t_idx] += perf
            perf_count[a_idx, t_idx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        perf_mean = np.where(perf_count > 0, perf_sum / perf_count, np.nan)

    valid = ~np.isnan(perf_mean)
    y_idx, x_idx = np.where(valid)
    all_y, all_x = np.meshgrid(np.arange(n_a), np.arange(n_t), indexing="ij")
    interp = griddata((x_idx, y_idx), perf_mean[valid], (all_x, all_y), method="linear")
    smooth = gaussian_filter(np.nan_to_num(interp, nan=np.nanmean(perf_mean[valid])),
                             sigma=1.5)
    smooth[perf_count < 1] = np.nan

    return dict(perf_mean=perf_mean, perf_count=perf_count, perf_smooth=smooth,
                arousal_bins=arousal_bins, time_bins=time_bins)


def optimal_trajectory(surface):
    """Fit the time-varying optimal arousal trajectory and its half-width.

    At each time bin the optimum is the performance-weighted mean arousal over
    sufficiently populated arousal bins, and the half-width is the weighted SD
    around it. Both series are then fitted with a cubic polynomial, weighted by
    how many observations each bin contributed.
    """
    perf_mean, perf_count = surface["perf_mean"], surface["perf_count"]
    arousal_bins, time_bins = surface["arousal_bins"], surface["time_bins"]
    n_t = perf_mean.shape[1]

    centers = (arousal_bins[:-1] + arousal_bins[1:]) / 2
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2

    opt = np.full(n_t, np.nan)
    std = np.full(n_t, np.nan)
    confidence = np.zeros(n_t)

    for t in range(n_t):
        valid = perf_count[:, t] >= MIN_SAMPLES_PER_BIN
        if not valid.any():
            continue
        w = perf_mean[valid, t] * perf_count[valid, t]
        w = w / w.sum()
        opt[t] = np.average(centers[valid], weights=w)
        std[t] = np.sqrt(np.average((centers[valid] - opt[t]) ** 2, weights=w))
        confidence[t] = perf_count[valid, t].sum()

    fit = confidence >= MIN_CONFIDENCE_FOR_FIT
    w_fit = confidence[fit] / confidence[fit].max()
    c_opt = np.polyfit(time_centers[fit], opt[fit], POLY_DEGREE, w=w_fit)
    c_std = np.polyfit(time_centers[fit], std[fit], POLY_DEGREE, w=w_fit)

    return dict(
        time_centers=time_centers,
        optimal=np.polyval(c_opt, time_centers),
        std=np.maximum(np.polyval(c_std, time_centers), MIN_STD),
        optimal_raw=opt, std_raw=std, confidence=confidence,
    )


# --------------------------------------------------------------------------- #
# Deviation metrics
# --------------------------------------------------------------------------- #

def deviation_metrics(trials_df, traj, band_mult=1.0, indices=None):
    """All four per-trial deviation metrics, in one pass over the trials.

    ``pct_in_band`` counts time at or below the upper bound, so excursions
    below the lower bound are treated as in-band; it is therefore exactly
    ``100 - pct_above``. ``excursion_rate`` counts above-band *bins* per
    minute, not contiguous runs, so it is likewise a rescaling of
    ``pct_above``. Only ``mean_excursion`` is independent of the other three,
    which matters when correcting for multiple comparisons across them.
    ``analyses/band_limbs.py`` separates the two sides of the band.
    """
    opt, std = traj["optimal"], traj["std"]
    T = len(opt)
    rows = trials_df if indices is None else trials_df.iloc[indices]

    out = []
    for _, row in rows.iterrows():
        binned = _binned_arousal(row["new_arousal"])
        above = below = inside = 0
        magnitudes = []
        for t_idx, w_ar in enumerate(binned[:T]):
            if np.isnan(opt[t_idx]) or np.isnan(std[t_idx]):
                continue
            upper = opt[t_idx] + std[t_idx] * band_mult
            lower = opt[t_idx] - std[t_idx] * band_mult
            if w_ar > upper:
                above += 1
                magnitudes.append(w_ar - upper)
            elif w_ar < lower:
                below += 1
            else:
                inside += 1
        total = above + below + inside
        if total:
            out.append({
                "subject": row["subject"],
                "performance": row["performance"] / FS,
                "pct_above": above / total * 100,
                "pct_in_band": (inside + below) / total * 100,
                "mean_excursion": float(np.mean(magnitudes)) if magnitudes else 0.0,
                "excursion_rate": above / (total / 60),
            })
    return pd.DataFrame(out)


def cohens_d_good_bad(df, col="pct_above", q=(25, 75)):
    """Effect size separating bottom- from top-quartile trials, and its p-value."""
    lo, hi = np.percentile(df["performance"], q)
    bad = df.loc[df["performance"] <= lo, col].to_numpy()
    good = df.loc[df["performance"] >= hi, col].to_numpy()
    if len(bad) < 3 or len(good) < 3:
        return np.nan, np.nan
    pooled = np.sqrt((bad.std() ** 2 + good.std() ** 2) / 2)
    if pooled == 0:
        return np.nan, np.nan
    return (bad.mean() - good.mean()) / pooled, ttest_ind(bad, good).pvalue


def band_width_sweep(trials_df, traj, indices, multipliers=BAND_MULTIPLIERS):
    """Cohen's d against band width, for one group of trials."""
    ds, ps = [], []
    for m in multipliers:
        d, p = cohens_d_good_bad(deviation_metrics(trials_df, traj, m, indices))
        ds.append(d)
        ps.append(p)
    return np.array(ds), np.array(ps)


# --------------------------------------------------------------------------- #
# Dwell times above band
# --------------------------------------------------------------------------- #

def dwell_times(trials_df, traj, band_mult=1.0, indices=None,
                min_sustain_s=2, merge_gap_s=1, low_perf_only=True):
    """Durations of sustained above-band excursions, in seconds.

    Short dips back into the band (``merge_gap_s`` or shorter) are bridged, and
    excursions shorter than ``min_sustain_s`` are discarded as transients
    inconsistent with a state change in LC-NE dynamics.
    """
    opt, std = traj["optimal"], traj["std"]
    T = len(opt)
    rows = trials_df if indices is None else trials_df.iloc[indices]

    if low_perf_only:
        p25 = np.percentile(rows["performance"].to_numpy() / FS, 25)

    dwells = []
    for _, row in rows.iterrows():
        if low_perf_only and row["performance"] / FS > p25:
            continue
        binned = _binned_arousal(row["new_arousal"])[:T]
        above = np.array([
            (not np.isnan(opt[i]) and not np.isnan(std[i])
             and v > opt[i] + std[i] * band_mult)
            for i, v in enumerate(binned)
        ])

        merged = above.copy()
        i = 0
        while i < len(merged):
            if not merged[i]:
                j = i
                while j < len(merged) and not merged[j]:
                    j += 1
                if (j - i) <= merge_gap_s and i > 0 and j < len(merged):
                    merged[i:j] = True
                i = j
            else:
                i += 1

        run = 0
        for v in np.append(merged, False):
            if v:
                run += 1
            else:
                if run >= min_sustain_s:
                    dwells.append(run)
                run = 0
    return np.array(dwells)


def sensitivity_score(df):
    """Arousal-sensitivity index: low parasympathetic tone, high cortical gamma."""
    return -df["hrv_z"] + df["gamma_z"]
