"""Signal-level helpers: artefact removal, band filtering, normalisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .config import BANDS, FS, N_EEG


def hampel_despike(x, fs=FS, window_sec=1.0, n_sigma=4.0):
    """Replace impulsive artefacts with interpolated values.

    Flags samples whose deviation from a rolling median exceeds ``n_sigma``
    robust sigmas (1.4826 x MAD), then linearly interpolates across them.
    """
    s = pd.Series(np.asarray(x, dtype=float))
    k = int(max(3, round(window_sec * fs)))

    med = s.rolling(k, center=True, min_periods=1).median()
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(k, center=True, min_periods=1).median()
    outliers = abs_dev > (n_sigma * 1.4826 * mad)

    clean = s.copy()
    clean[outliers] = np.nan
    return clean.interpolate(limit_direction="both").to_numpy()


def extract_bandpowers_timeseries(X, sf=FS):
    """Band-pass the EEG channels into theta/alpha/beta/gamma time series.

    ``X`` is (trials, channels, samples). The first ``N_EEG`` channels are
    filtered into each band and stacked, giving 64 x 4 = 256 channels; any
    remaining channels are appended unfiltered.
    """
    _, channels, _ = X.shape

    filtered = []
    for low, high in BANDS.values():
        sos = butter(4, [low / (sf / 2), high / (sf / 2)], btype="band", output="sos")
        filtered.append(sosfiltfilt(sos, X[:, :N_EEG, :], axis=-1))

    stacked = np.concatenate(filtered, axis=1)
    if channels > N_EEG:
        return np.concatenate([stacked, X[:, N_EEG:, :]], axis=1)
    return stacked


def zscore_per_subject(ring_df):
    """Z-score each channel within subject, over all that subject's epochs.

    Statistics are pooled across epochs and time, giving one mean/std per
    (subject, channel).
    """
    out = ring_df.copy()
    for subj in out["subj_idx"].unique():
        mask = out["subj_idx"] == subj
        arr = np.stack(out.loc[mask, "data"].to_numpy())      # (N, C, T)
        mean = arr.mean(axis=(0, 2), keepdims=True)
        std = arr.std(axis=(0, 2), keepdims=True) + 1e-8
        normed = (arr - mean) / std
        for i, idx in enumerate(out.index[mask]):
            out.at[idx, "data"] = normed[i]
    return out


def normalize_train_val(X_train, X_val):
    """Z-score using training statistics only.

    ``X_val`` may be an ndarray or a list of variable-length (C, T) arrays.
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8

    X_train_n = (X_train - mean) / std
    m, s = mean[0], std[0]                      # (C, 1) broadcast over time
    if isinstance(X_val, list):
        X_val_n = [(x - m) / s for x in X_val]
    else:
        X_val_n = (X_val - mean) / std
    return X_train_n, X_val_n


def asymmetric_iir(update_samples=16, tau_rise=0.75, tau_fall=1.5, fs=FS):
    """Smoothing coefficients for the continuous arousal index.

    Arousal is allowed to rise faster than it falls, matching sympathetic
    activation latency (<1 s) against parasympathetic recovery (1-3 s).
    Returns ``(alpha_up, alpha_down)``.
    """
    dt = update_samples / fs
    return 1 - np.exp(-dt / tau_rise), 1 - np.exp(-dt / tau_fall)
