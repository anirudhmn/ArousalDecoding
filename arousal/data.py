"""Loading ring epochs and assembling the trial-level analysis table."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from . import config as cfg
from .config import (CALIBRATION_CONDITION, CH, RING_EVENTS,
                     RING_EVENTS_ONLINE, RING_LABEL, TRIAL_EVENTS)
from .signals import extract_bandpowers_timeseries, hampel_despike

# Raw channels carried alongside the decoded arousal, for external validation.
RAW_METRICS = {
    "hr_signal": CH["HR"],
    "hrv_signal": CH["HRV-pNN35"],
    "eda_phasic": CH["EDA-phasic"],
    "eda_tonic": CH["EDA-tonic"],
    "pupil_signal": [CH["PUP-L"], CH["PUP-R"]],   # averaged
    "bci_arousal": CH["BCI-raw"],
    "old_arousal": CH["FB-HB-0-1-nrm"],           # Faller et al. decoder output
}


def load_ring_epochs(path=RING_EVENTS):
    """Fixed-length (144, 512) ring epochs, with task-demand labels attached."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    df = pd.DataFrame(payload["events"] if isinstance(payload, dict) else payload)
    df = df.sort_values(["subj_idx", "trial_idx", "ring_idx"]).reset_index(drop=True)
    df["ring_position"] = df.groupby(["subj_idx", "trial_idx"]).cumcount()
    df["label"] = df["ring_size"].map(RING_LABEL).fillna(-1).astype(int)
    return df


def load_ring_epochs_online(path=RING_EVENTS_ONLINE):
    """Variable-length ring epochs; concatenated into continuous trials."""
    df = pd.DataFrame(pd.read_pickle(path))
    df = df.sort_values(["subj_idx", "trial_idx", "ring_idx"]).reset_index(drop=True)
    df["label"] = df["ring_size"].map(RING_LABEL).fillna(-1).astype(int)
    return df


FEATURE_SETS = ("physio", "eeg", "all")


def prepare_features(ring_df, feature_set):
    """Select the input channels for one decoder configuration.

    Returns ``(df, modalities)`` where ``df['data']`` holds only the channels
    that configuration uses, already in the order the modality groups expect:

    ``physio``  the 7 peripheral channels
    ``eeg``     64 channels x 4 bands = 256 band-passed EEG channels
    ``all``     the 256 EEG channels followed by the 7 peripheral channels
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}")

    out = ring_df.copy()
    if feature_set == "physio":
        out["data"] = [x[cfg.PERIPHERAL_CHANNELS, :] for x in out["data"]]
        return out, cfg.MODALITIES_PERIPHERAL

    n_bands = 4 * cfg.N_EEG                     # 256
    # Band-passing returns the 256 EEG band channels followed by the original
    # non-EEG channels, so peripheral channel c lands at n_bands + (c - 64).
    periph = [n_bands + (c - cfg.N_EEG) for c in cfg.PERIPHERAL_CHANNELS]
    keep = list(range(n_bands)) if feature_set == "eeg" else list(range(n_bands)) + periph

    out["data"] = [extract_bandpowers_timeseries(x[None, ...])[0][keep, :]
                   for x in out["data"]]
    return out, (cfg.MODALITIES_EEG if feature_set == "eeg" else cfg.MODALITIES_ALL)


def concatenate_trials(subj_df, channels=None):
    """Concatenate a subject's rings into whole-trial (C, T) arrays.

    Returns ``(trials, labels, meta)`` where ``meta`` is a DataFrame with one
    row per trial carrying ``trial_idx``, ``condition`` and ``difficulty``.
    Trial order matches the sorted ``trial_idx``, which is the order the
    decoded-arousal results are stored in.
    """
    trials, labels, meta = [], [], []
    for trial_idx, g in subj_df.groupby("trial_idx", sort=True):
        g = g.sort_values("ring_idx")
        data = np.concatenate(
            [r[channels, :] if channels is not None else r for r in g["data"]], axis=1)
        lab = np.concatenate([np.full(r.shape[1], l)
                              for r, l in zip(g["data"], g["label"])])
        trials.append(data)
        labels.append(lab)
        meta.append({
            "trial_idx": trial_idx,
            "condition": g["condition"].iloc[-1],
            "difficulty": 1 if g["trial_difficulty"].iloc[-1] == "hard" else 0,
        })
    return trials, labels, pd.DataFrame(meta)


def _raw_metric_traces(subj_df):
    """Per-trial raw physiological traces, min-max scaled to 0-100.

    Each channel is scaled within its own trial. This removes between-trial
    differences in absolute level, so the traces are comparable in shape but not
    in level. ``analyses/scaling_parity.py`` quantifies what that costs when
    these columns are compared against the decoded index, which is scaled per
    subject instead.
    """
    names = list(RAW_METRICS)
    out = {}
    for trial_idx, g in subj_df.groupby("trial_idx", sort=True):
        g = g.sort_values("ring_idx")
        parts = []
        for _, row in g.iterrows():
            d = row["data"]
            vals = np.zeros((len(names), d.shape[1]))
            for i, name in enumerate(names):
                ch = RAW_METRICS[name]
                vals[i] = hampel_despike(d[ch].mean(0)) if isinstance(ch, list) else d[ch]
            parts.append(vals)
        y = np.concatenate(parts, axis=1)
        lo = y.min(axis=1, keepdims=True)
        hi = y.max(axis=1, keepdims=True)
        out[trial_idx] = (y - lo) / (hi - lo + 1e-8) * 100
    return names, out


def build_trial_table(arousal_results, ring_online=None, trial_events_path=TRIAL_EVENTS):
    """Assemble the trial-level table that every downstream analysis uses.

    ``arousal_results`` is the ``final_results`` list from the online decoding
    stage: ``arousal_results[s][t]`` is (n_repetitions, T).

    One row per closed-loop trial, with the decoded arousal trace, the raw
    physiological traces, performance (flight time in samples) and the trial's
    condition and difficulty.
    """
    if ring_online is None:
        ring_online = load_ring_epochs_online()

    trial_events = pd.DataFrame(pd.read_pickle(trial_events_path))
    closed_loop = ring_online[ring_online["condition"] != CALIBRATION_CONDITION]
    subjects = sorted(ring_online["subj_idx"].unique())

    rows = []
    for s, subj_id in enumerate(subjects):
        subj_df = closed_loop[closed_loop["subj_idx"] == subj_id]
        names, raw = _raw_metric_traces(subj_df)
        _, _, meta = concatenate_trials(subj_df)

        for t, trial_array in enumerate(arousal_results[s]):
            arousal_trace = np.asarray(trial_array).mean(axis=0)
            mean_arousal = float(arousal_trace.mean())
            if mean_arousal == 0:
                continue                       # trial too short to decode

            m = meta.iloc[t]
            traces = raw[m["trial_idx"]]

            # Look the events up by the trial's real index, not its position
            # among closed-loop trials.
            ev = trial_events.loc[(trial_events["subj_idx"] == subj_id)
                                  & (trial_events["trial_idx"] == m["trial_idx"]),
                                  "events"]

            row = {
                "subject": s,
                "subj_id": subj_id,
                "trial": t,
                "trial_idx": m["trial_idx"],
                "difficulty": m["difficulty"],
                "condition": m["condition"],
                "performance": arousal_trace.shape[0],   # flight time, samples
                "arousal": mean_arousal,
                "new_arousal": arousal_trace,
                "events": ev.iloc[0] if len(ev) else None,
            }
            row.update({name: traces[i] for i, name in enumerate(names)})
            rows.append(row)

    return pd.DataFrame(rows)


def baseline_features(ring_online=None):
    """Per-subject baseline physiology from the calibration block.

    Band-limited EEG power and mean peripheral levels, averaged over every
    condition-0 ring epoch for that subject.
    """
    if ring_online is None:
        ring_online = load_ring_epochs_online()

    df = ring_online[ring_online["condition"] == CALIBRATION_CONDITION]
    subjects = sorted(df["subj_idx"].unique())

    rows = []
    for s, subj_id in enumerate(subjects):
        g = df[df["subj_idx"] == subj_id]
        banded = np.concatenate(
            [extract_bandpowers_timeseries(x[None, ...])[0] for x in g["data"]], axis=1)
        rows.append({
            "subject": s,
            "subj_id": subj_id,
            "theta": banded[0:64].mean(),
            "alpha": banded[64:128].mean(),
            "beta": banded[128:192].mean(),
            "gamma": banded[192:256].mean(),
            "hr": banded[256 + CH["HR"] - 64].mean(),
            "hrv": banded[256 + CH["HRV-pNN35"] - 64].mean(),
            "resp": banded[256 + CH["RESP"] - 64].mean(),
            "eda_p": banded[256 + CH["EDA-phasic"] - 64].mean(),
            "eda_t": banded[256 + CH["EDA-tonic"] - 64].mean(),
            "pup": banded[[256 + CH["PUP-L"] - 64, 256 + CH["PUP-R"] - 64]].mean(),
        })
    return pd.DataFrame(rows)


BASELINE_COLS = ["theta", "alpha", "beta", "gamma", "hr", "hrv", "resp",
                 "eda_p", "eda_t", "pup"]


def attach_baselines(trial_df, baselines):
    """Merge baseline features onto the trial table and z-score them.

    The z-scores are taken over the merged *trial-level* table, so subjects are
    weighted by how many trials they completed. See METHODS.md for why that
    convention is kept rather than z-scoring the 16 per-subject values.
    """
    merged = trial_df.merge(baselines.drop(columns=["subj_id"]),
                            on="subject", how="left")
    for c in BASELINE_COLS:
        merged[c + "_z"] = (merged[c] - merged[c].mean()) / merged[c].std()
    return merged
