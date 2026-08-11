"""Channel maps, experiment constants and canonical paths.

Everything that describes *the dataset* rather than *an analysis* lives here, so
that the notebooks never hard-code a channel index.
"""

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = DATA / "results"
FIGURES = ROOT / "figures"
ASSETS = ROOT / "assets"      # static schematics used in figures

# Fixed-length (144, 512) ring epochs, all conditions. Used for decoder training.
RING_EVENTS = DATA / "ring_events.pkl"
# Variable-length ring epochs, all conditions. Concatenated into continuous
# per-trial traces for sliding-window arousal decoding.
RING_EVENTS_ONLINE = DATA / "ring_events_online.pkl"
# Per-trial list of (ring_size, sample_index) crossings.
TRIAL_EVENTS = DATA / "trial_events.pkl"

# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #

FS = 256                # Hz, after downsampling from the 2048 Hz BioSemi rate
EPOCH_SAMPLES = 512     # 2 s pre-crossing epoch

# The 16 subjects retained for analysis (S13-S16 were excluded upstream).
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20]

# Feedback condition codes, verbatim from the Faller et al. (2019) README.
# Note that condition 2 still delivers 50% veridical BCI feedback.
CONDITIONS = {
    0: "silence-openloop",   # calibration block, used for decoder training
    1: "silence",            # closed loop, no feedback
    2: "half",               # closed loop, 50% sham / 50% BCI
    3: "bci",                # closed loop, full BCI feedback
}
CALIBRATION_CONDITION = 0
CONTROL_CONDITIONS = (1, 2)   # "control" trials for the optimal-trajectory fit
FEEDBACK_CONDITION = 3

# --------------------------------------------------------------------------- #
# Channel layout of the raw (144, T) epoch arrays
# --------------------------------------------------------------------------- #

CHANNEL_GROUPS = {
    "EEG": [
        "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
        "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5",
        "P7", "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz",
        "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6",
        "FC4", "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4",
        "CP2", "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
    ],
    "Cardiac": ["ECG", "HR", "HR-delta", "HR-delta-abs", "HRV-pNN35"],
    "Respiration": ["RESP"],
    "Electrodermal Activity": ["EDA", "EDAz", "EDA-phasic", "EDA-SMNA", "EDA-tonic"],
    "Pupillometry & Eye Tracking": [
        "PUP-L", "PUP-R", "PORX", "PORY", "PORX-L", "PORY-L", "PORX-R", "PORY-R",
        "EYE-Time", "EYE-Validity", "PUP-int-L", "PUP-int-R", "PORX-int",
        "PORY-int", "PORX-int-L", "PORY-int-L", "PORX-int-R", "PORY-int-R",
        "PUP-conv05-4-L", "PUP-conv05-4-R", "PUP-logbp06-L", "PUP-logbp06-R",
    ],
    "Eye Movement Events": [
        "Sac-Amp-Avg", "Sac-Angle-Abs-Avg", "Sac-Dur-Avg", "Sac-VMax-Avg",
        "Fix-Dur-Avg", "Sac-rate-pmin", "Fix-rate-pmin", "Blink-rate-pmin",
    ],
    "Joystick Controls": [
        "Joy-Pitch", "Joy-Roll", "Joy-Pitch-acc", "Joy-Roll-acc", "Joy-Pitch-pw",
        "Joy-Roll-pw", "Joy-Pitch-pw016-3", "Joy-Roll-pw016-3",
        "Joy-Pitch-pw3plus", "Joy-Roll-pw3plus", "Joy-Pitch-pw0016",
        "Joy-Roll-pw0016",
    ],
    "Neurofeedback Signals": [
        "BCI-raw", "FB-HB-0-1", "FB-HB-0-1-nrm", "FB-HB-raw", "FB-Mix-fact",
        "Shm-2-2", "Shm-raw",
    ],
    "Flight Trajectory & Path Metrics": [
        "Plane-pos-len", "Plane-pos-height", "Path-steps", "Path-low",
        "Path-high", "Path-width", "Path-avg", "Path-avg-2s-smooth",
    ],
    "Head Motion": [
        "Head-raw-x", "Head-raw-y", "Head-raw-z", "Head-x", "Head-y", "Head-z",
    ],
    "Metadata & Performance": [
        "Condition", "Flight-time", "Course-type", "FB-type", "Ring-type",
        "LSL-time",
    ],
}

FEATURE_NAMES = [name for group in CHANNEL_GROUPS.values() for name in group]
CH = {name: i for i, name in enumerate(FEATURE_NAMES)}

N_EEG = len(CHANNEL_GROUPS["EEG"])          # 64

# The seven peripheral channels that make up the autonomic decoder input.
PERIPHERAL_CHANNELS = [
    CH["HR"], CH["HRV-pNN35"], CH["RESP"],
    CH["EDA-phasic"], CH["EDA-tonic"], CH["PUP-L"], CH["PUP-R"],
]
PERIPHERAL_NAMES = ["HR", "HRV-pNN35", "RESP", "EDA-phasic", "EDA-tonic",
                    "PUP-L", "PUP-R"]

EEG_CHANNELS = list(range(N_EEG))

# --------------------------------------------------------------------------- #
# Modality groupings passed to the model, indexed into the *selected* channels
# --------------------------------------------------------------------------- #

# Autonomic-only decoder: input is X[PERIPHERAL_CHANNELS], so 7 channels.
MODALITIES_PERIPHERAL = {
    "HR": [0, 1],      # HR, HRV-pNN35
    "RESP": [2],
    "EDA": [3, 4],     # phasic, tonic
    "Pupil": [5, 6],   # left, right
}

# EEG-only decoder: 64 channels x 4 bands = 256, one modality per band.
MODALITIES_EEG = {
    "theta": list(range(0, 64)),
    "alpha": list(range(64, 128)),
    "beta": list(range(128, 192)),
    "gamma": list(range(192, 256)),
}

# Combined decoder: 256 band-filtered EEG channels followed by the 7 peripheral.
MODALITIES_ALL = {
    "EEG": list(range(0, 256)),
    "HR": [256, 257],
    "RESP": [258],
    "EDA": [259, 260],
    "Pupil": [261, 262],
}

# EEG bands, in Hz. Used both for decoder input and for the baseline gamma
# feature that drives the sensitivity score.
BANDS = {
    "theta": (4, 7),
    "alpha": (8, 15),
    "beta": (16, 32),
    "gamma": (32, 55),
}

# --------------------------------------------------------------------------- #
# Ring difficulty labels
# --------------------------------------------------------------------------- #

# Following Faller et al., large rings are low task demand (class 0), medium and
# small rings are high task demand (class 1).
RING_LABEL = {"large": 0, "medium": 1, "small": 1}


def peripheral_view(x: np.ndarray) -> np.ndarray:
    """Select the 7 peripheral channels from a (144, T) epoch."""
    return x[PERIPHERAL_CHANNELS, :]
