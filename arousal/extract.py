"""Turn the raw Faller et al. .mat recordings into ring-locked epochs.

Two epoch sets are produced from the same event log:

``ring_events.pkl``
    Fixed 2 s (512-sample) windows ending at each ring crossing. Used to train
    and cross-validate the decoder.
``ring_events_online.pkl``
    Variable-length windows spanning ring *n-1* to ring *n*, so that a trial's
    epochs concatenate back into one continuous recording. Used for
    sliding-window arousal decoding.
"""

from __future__ import annotations

import os
import pickle
import re

import numpy as np
import scipy.io
from scipy.io.matlab import mat_struct

from .config import EPOCH_SAMPLES, PERIPHERAL_CHANNELS, SUBJECTS, N_EEG

# Ring diameter -> (size, course difficulty). The easy and hard courses use
# different absolute diameters for the same nominal size.
RINGSIZE_MAP = {
    108: "large_easy", 72: "medium_easy", 36: "small_easy",
    60: "large_hard", 30: "medium_hard", 14.3999: "small_hard",
}
START_LABELS = {"F_CL_Sil": "START-F_CL_Sil_50_100", "C_OLoop": "START-C_OLoop"}
RING_PATTERN = re.compile(r"^RingPassed_Size_([0-9]+(?:\.[0-9]+)?)_Cond_([0-9]+)$")

# Baseline z-scoring is applied to the EEG and peripheral channels only; the
# remaining channels (feedback signals, joystick, head motion) stay raw.
NORMALISED_CHANNELS = list(range(N_EEG)) + PERIPHERAL_CHANNELS


# --------------------------------------------------------------------------- #
# MATLAB loading
# --------------------------------------------------------------------------- #

def _matobj_to_dict(obj):
    return {f: _parse(getattr(obj, f)) for f in obj._fieldnames}


def _parse(val):
    if isinstance(val, mat_struct):
        return _matobj_to_dict(val)
    if isinstance(val, np.ndarray) and val.dtype == "O":
        parsed = [_matobj_to_dict(e) if isinstance(e, mat_struct) else e for e in val]
        if not parsed:
            return []
        return parsed[0] if len(parsed) == 1 else parsed
    return val


def loadmat_recursive(filename):
    """Load a .mat file, unpacking nested mat_structs into dicts."""
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return {k: _parse(v) for k, v in data.items()}


def load_recording(path):
    return loadmat_recursive(path)["actualVariable"]["EEG_full"]


# --------------------------------------------------------------------------- #
# Epoching
# --------------------------------------------------------------------------- #

def baseline_stats(path):
    """Per-channel mean and SD from a resting eyes-open recording."""
    mat = load_recording(path)
    data = mat["data"][NORMALISED_CHANNELS]
    means = np.nanmean(data, axis=1)[:, None]
    stds = np.nanstd(data, axis=1)
    stds = np.where(stds == 0, 1.0, stds)[:, None]
    return means, stds


def _trial_intervals(events, start_label):
    """Pair each trial-start marker with the next boundary (crash) marker."""
    starts = sorted(t for label, t in events if label == start_label)
    ends = sorted(t for label, t in events if label == "boundary")
    ends.append(events[-1][1])

    intervals, it = [], iter(ends)
    current = next(it, None)
    for start in starts:
        while current is not None and current < start:
            current = next(it, None)
        if current is None:
            break
        intervals.append((start, current))
    return intervals


def parse_ring_events(path, baselines=None, online=False):
    """Epoch one recording around every ring crossing.

    With ``online=False`` each epoch is the fixed 2 s window ending at the
    crossing. With ``online=True`` each epoch runs from the previous crossing
    to the current one, so a trial's epochs tile it without gaps.
    """
    subj_idx = int(os.path.basename(path)[1:3])
    mat = load_recording(path)
    if baselines is not None:
        mat["data"][NORMALISED_CHANNELS] = (
            (mat["data"][NORMALISED_CHANNELS] - baselines[0]) / baselines[1])

    events = [(e["type"], e["latency"]) for e in mat["event"]]
    key = next(k for k in START_LABELS if k in os.path.basename(path))
    intervals = _trial_intervals(events, START_LABELS[key])

    rings = [(t, *m.groups()) for label, t in events
             if (m := RING_PATTERN.match(label)) and "ECG" not in label]

    epochs = []
    for trial_idx, (t_start, t_end) in enumerate(intervals):
        trial_rings = [r for r in rings if t_start < r[0] < t_end]
        if not trial_rings:
            continue
        # Close the trial with a synthetic crossing at the boundary event, so
        # the final segment up to the crash is kept.
        trial_rings.append((t_end, trial_rings[-1][1], trial_rings[-1][2]))

        end_idx = 0
        for i, (curr, size_code, cond_code) in enumerate(trial_rings):
            start_idx = round(t_start) if i == 0 else round(end_idx)
            end_idx = round(curr)
            if not online:
                start_idx = round(max(0, end_idx - EPOCH_SAMPLES))
                end_idx = round(min(mat["data"].shape[1], end_idx))

            size, difficulty = RINGSIZE_MAP[float(size_code)].split("_")
            epochs.append({
                "subj_idx": subj_idx,
                "trial_idx": trial_idx,
                "ring_idx": i + 1,
                "condition": int(cond_code),
                "trial_difficulty": difficulty,
                "ring_size": size,
                "n_samples": end_idx - start_idx,
                "data": mat["data"][:, start_idx:end_idx],
            })
    return epochs


def parse_trial_events(path):
    """One row per trial, listing (ring_size, sample_within_trial) crossings."""
    subj_idx = int(os.path.basename(path)[1:3])
    mat = load_recording(path)
    events = [(e["type"], e["latency"]) for e in mat["event"]]
    key = next(k for k in START_LABELS if k in os.path.basename(path))
    intervals = _trial_intervals(events, START_LABELS[key])

    rings = [(t, *m.groups()) for label, t in events
             if (m := RING_PATTERN.match(label)) and "ECG" not in label]

    rows = []
    for trial_idx, (t_start, t_end) in enumerate(intervals):
        trial_rings = [r for r in rings if t_start < r[0] < t_end]
        if not trial_rings:
            continue
        listed, difficulty = [], None
        for curr, size_code, _ in trial_rings:
            size, difficulty = RINGSIZE_MAP[float(size_code)].split("_")
            listed.append((size, int(round(curr - t_start))))
        rows.append({"subj_idx": subj_idx, "trial_idx": trial_idx,
                     "trial_difficulty": difficulty, "events": listed})
    return rows


# --------------------------------------------------------------------------- #
# Whole-cohort drivers
# --------------------------------------------------------------------------- #

def subject_files(base_dir, sidx):
    return {
        "baseline": os.path.join(base_dir, f"S{sidx:02d}_B_RWEO_PreOL.mat"),
        "open_loop": os.path.join(base_dir, f"S{sidx:02d}_C_OLoop.mat"),
        "closed_loop": os.path.join(base_dir, f"S{sidx:02d}_F_CL_Sil_50_100.mat"),
    }


def build_ring_events(base_dir, online=False, subjects=SUBJECTS, verbose=True):
    """Epoch every subject, dropping any epoch with NaNs in a modelled channel."""
    all_epochs, open_lengths = [], []
    for sidx in subjects:
        f = subject_files(base_dir, sidx)
        stats = baseline_stats(f["baseline"])
        open_loop = parse_ring_events(f["open_loop"], stats, online)
        closed_loop = parse_ring_events(f["closed_loop"], stats, online)
        all_epochs += open_loop + closed_loop
        open_lengths.append(len(open_loop))
        if verbose:
            print(f"  S{sidx:02d}: {len(open_loop)} open-loop + "
                  f"{len(closed_loop)} closed-loop epochs")

    keep = [e for e in all_epochs
            if not np.isnan(e["data"][NORMALISED_CHANNELS]).any()]
    if verbose:
        print(f"  dropped {len(all_epochs) - len(keep)} epochs containing NaNs")
    return keep, open_lengths


def build_trial_events(base_dir, subjects=SUBJECTS):
    rows = []
    for sidx in subjects:
        rows += parse_trial_events(subject_files(base_dir, sidx)["closed_loop"])
    return rows


def write_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
