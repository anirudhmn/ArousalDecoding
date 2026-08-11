"""Multimodal physiological arousal decoding in a boundary-avoidance task.

Re-analysis of Faller et al. (2019), PNAS 116:6482-6490.
IEEE DataPort: https://doi.org/10.21227/rn3e-bp31

Layout
------
``config``    channel maps, condition codes, paths
``extract``   raw .mat recordings -> ring-locked epochs
``signals``   filtering, despiking, normalisation
``models``    the multimodal decoder
``training``  cross-validation and continuous arousal inference
``data``      epoch loading and the trial-level analysis table
``yerkes``    quadratic mixed-effects arousal-performance models
``control``   optimal trajectory, deviation metrics, control bands
``plotting``  shared figure style
"""

from . import (config, control, data, extract, models, plotting, signals,
               training, yerkes)

__all__ = ["config", "control", "data", "extract", "models", "plotting",
           "signals", "training", "yerkes"]
__version__ = "1.0.0"
