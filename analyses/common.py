"""Shared setup for the analysis scripts in this folder.

Puts the repository root on the import path so that ``arousal`` resolves from a
plain checkout, and creates the ``results/`` folder that every script writes to.
The package is imported read-only. Nothing here modifies it.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = pathlib.Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

import numpy as np
import pandas as pd

from arousal import config as cfg, control as C, data as D, yerkes as Y
from arousal.config import FS

# Note on the name ``C``. Patsy resolves the categorical marker ``C()`` in a
# formula from the calling namespace, so a module that does ``from common
# import C`` and then calls ``smf.ols("y ~ C(x)", ...)`` will find the control
# module instead and raise. ``smf.mixedlm`` is unaffected because it evaluates
# formulas in a clean namespace, which is why every model here is a mixed model.
# Import the control module under another name if you need plain OLS.


def trial_table():
    """The cached trial-level table written by notebook 03."""
    return pd.read_pickle(ROOT / "data" / "trial_table.pkl")


def hard_control(df):
    """Hard-course trials under silence and half-sham feedback.

    These are the control trials that the optimal arousal trajectory is fitted
    on. Condition 2 delivers 50 percent veridical feedback, so it is a partial
    rather than a pure control; ``pipeline_sensitivity.py`` quantifies what
    including it costs.
    """
    return df[(df.difficulty == 1) & (df.condition.isin(cfg.CONTROL_CONDITIONS))
              ].reset_index(drop=True)


def subject_scores(df):
    """One arousal-sensitivity score per subject, from baseline HRV and gamma.

    Retained so that the original grouping can be reproduced. ``gamma_z`` is the
    mean of the signed band-passed signal, which is close to zero by
    construction; ``gamma_definitions.py`` shows that the interaction it
    supports does not hold under any genuine power measure, and
    ``vagal_composite.py`` and ``calibration_profile.py`` develop the
    replacements.
    """
    per = df.groupby("subject")[["hrv_z", "gamma_z"]].first()
    return (-per["hrv_z"] + per["gamma_z"]).rename("score")
