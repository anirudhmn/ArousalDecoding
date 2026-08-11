# Methods notes

Conventions that are load-bearing but not obvious from reading the code. Nothing
here changes a result. It records what the numbers mean.

## Signal conventions

**Decoder warm-up.** Sliding-window inference starts at sample 256, so the first
second of every arousal trace is exactly zero. The IIR filter then ramps up from
zero over the next second or two. Trial-mean arousal includes that warm-up
segment. Performance is trial duration, so the warm-up occupies a larger
fraction of a short trial than of a long one.
`analyses/pipeline_sensitivity.py` quantifies the effect of trimming it.

**The two arousal signals are scaled differently.** The peripheral decoder
output is rescaled per subject against the 5th and 95th percentiles of that
subject's training distribution and clipped to 0-100, which preserves
trial-to-trial differences in level. The comparison signal `FB-HB-0-1-nrm`, the
normalised feedback channel driven by the Faller et al. EEG decoder, is min-max
scaled within each trial in `data._raw_metric_traces`. Every trial then spans
exactly 0-100 and its mean reflects trace shape rather than arousal level. The
same per-trial scaling applies to the raw HR, HRV, EDA and pupil traces.

This asymmetry matters whenever the two signals are compared at trial level.
`analyses/scaling_parity.py` refits the comparison with both signals on each
scaling in turn.

**The 0-100 index is subject-specific and clipped.** Because the bounds come
from that subject's own training distribution, index values are comparable
within a subject but not in absolute terms across subjects. Clipping is applied
before the IIR filter.

**Baseline normalisation covers only some channels.** During extraction, epochs
are z-scored against the subject's resting recording for the 64 EEG and 7
peripheral channels only. Feedback, joystick, head-motion and metadata channels
stay in raw units.

## Feature conventions

**The baseline gamma feature is a filtered-signal mean, not a power.**
`data.baseline_features` averages the band-pass filtered signal itself, not its
square or its envelope. A band-passed signal has a near-zero mean, so the raw
values are around 1e-4 and the z-scores are dominated by small residual offsets.
The values are deterministic and subject-specific, but the quantity is not band
power in the usual sense. The same applies to theta, alpha and beta.
`analyses/gamma_definitions.py` recomputes the feature five ways and shows that
the moderation it supports does not survive any genuine power measure.

**Baseline z-scores are trial-weighted.** `data.attach_baselines` z-scores the
baseline features over the merged trial-level table, so subjects contribute in
proportion to how many trials they completed rather than once each. The median
split in notebook 04 is likewise taken over trials. It gives 8 subjects per
group here. The convention is kept because every stored result depends on it;
`analyses/baseline_marker_screen.py` works from per-subject values instead.

## Metric conventions

**Three of the four deviation metrics are the same measurement.**
`pct_in_band` is exactly `100 - pct_above`, because a bin counts as in band
whenever it is at or below the upper bound, so excursions below the lower bound
count as in band too. `excursion_rate` counts above-band bins per minute, so it
is `pct_above` rescaled by trial length. Only `mean_excursion` is independent.
Correcting the four as if they were independent tests is therefore conservative
to the point of being misleading. `analyses/band_limbs.py` separates the two
sides of the band.

**Control trials include half-sham.** Condition 2 delivers a 50/50 blend of real
decoder output and an autoregressive surrogate, so it is not a pure control. The
optimal trajectory is fitted on conditions 1 and 2 together, giving 113
hard-course trials. `analyses/pipeline_sensitivity.py` refits on condition 1
alone.

**Band multipliers are chosen and evaluated on the same trials** in notebook 04.
The grid search maximises Cohen's d within each sensitivity group and then
reports that same d. `analyses/loso_control_bands.py` provides the
leave-one-subject-out version, and `analyses/heldout_full_bci.py` applies the
fitted parameters to a feedback condition that entered no part of the fit.

**Mixed models are fitted by maximum likelihood** (`reml=False`) throughout, not
by restricted maximum likelihood. This is required because AIC is compared
across models with different fixed-effect structures.

## Statistical conventions

**Clustering.** Each subject contributes many trials. Trial-level correlations
are reported as descriptive quantities only, and every inferential statement
uses a mixed model with a random intercept per subject. See
`analyses/clustered_statistics.py`.

**Multiple comparisons.** `analyses/multiplicity.py` states one policy and
applies it to every inferential p-value the analyses produce. Holm within
family; a family is a set of tests addressing one question with interchangeable
outcomes; pre-specified primary tests form a family of one and are labelled as
such; descriptive quantities enter no family.

**Cluster-based permutation.** All parameters are fixed in
`analyses/cluster_permutation.py` and printed with its results. The first two
time bins are dropped because the first second of every trace is exactly zero by
construction and the IIR filter ramps through the second, so those bins have no
between-subject variance.

## Reproducibility

Model fitting is seeded per subject, repetition and fold, so a rerun on the same
hardware reproduces itself. Everything downstream of the decoder is
deterministic given the decoder outputs.

Results stored from a different accelerator will not match bitwise. Retraining
the within-subject cross-validation on Apple Silicon reproduces stored CUDA AUC
values to within about 0.15 AUC points, and continuous decoding reproduces
stored traces at a per-trial correlation of about 0.996. Statistical agreement
rather than bitwise agreement is the standard to expect.

Approximate cost on an Apple M3 Max, using MPS for training and CPU for
attribution:

| stage | time |
|---|---|
| within-subject cross-validation, peripheral, 16 subjects, 10 repetitions | about 2.5 h |
| continuous decoding, 16 subjects, 10 repetitions | about 25 min |
| leave-one-subject-out decoder, both feature sets | about 1.5 h |
| label controls, 6 conditions, 16 subjects | about 1.5 h |
| cluster permutation, 10,000 permutations plus sensitivity simulation | about 5 min |
| the three analysis notebooks | under a minute each |
| every other analysis script | seconds to a couple of minutes |

CPU-only training of the full grid is roughly ten times slower than MPS and is
not recommended. The analysis notebooks need no accelerator at all.
