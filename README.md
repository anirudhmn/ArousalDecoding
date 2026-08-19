# Multimodal physiological arousal decoding in a boundary-avoidance task

Analysis code for a re-analysis of the publicly available dataset from Faller,
Cummings, Saproo and Sajda (2019), *PNAS* **116**:6482-6490.

Sixteen participants flew a virtual-reality boundary-avoidance task through
sequences of shrinking rings while receiving EEG-decoded auditory
neurofeedback, sham feedback, or silence. This code trains a multimodal decoder
on peripheral physiology, turns its output into a continuous arousal index,
tests that index against task performance and task phase, and asks what a
closed-loop controller built on it would need.

Every analysis here is offline. The peripheral decoder was never placed in the
feedback loop.

## Install

```bash
pip install -e .
```

Python 3.10 or later, PyTorch 2.0 or later. A GPU is optional. The code uses
CUDA if present and falls back to CPU.

## Data

No recordings are distributed with this repository. Download them from
[IEEE DataPort](https://doi.org/10.21227/rn3e-bp31) and point `RAW_DIR` in
notebook 00 at the folder of `.mat` files. Everything else is derived.

Notebook 00 writes three intermediates into `data/`:

| file | contents |
|---|---|
| `ring_events.pkl` | fixed 2 s (512-sample) epochs ending at each ring crossing |
| `ring_events_online.pkl` | variable-length epochs that tile each trial end to end |
| `trial_events.pkl` | per-trial ring-crossing times |

The two epoch files are around 4 GB each. Notebook 01 writes decoder outputs
into `data/results/`. Notebook 03 caches the trial-level table as
`data/trial_table.pkl`, which every analysis script reads.

## Running

### Notebooks

Numbered in dependency order.

| notebook | produces |
|---|---|
| `00_extract_epochs` | the three epoch files above |
| `01_train_decoders` | `results_offline_simple_{physio,eeg,all}.pkl`, `results_online_simple_physio.pkl` |
| `02_decoder_performance` | decoder summary tables and the trial-level AUC comparison |
| `03_yerkes_dodson` | the trial table used by everything downstream |
| `04_control_bands` | the optimal trajectory and the deviation-metric table |

Notebooks 00 and 01 are expensive and only need to run once. The three analysis
notebooks run in well under a minute each from the cached intermediates.

### Figures

Every figure in the paper is built by one script. Figure 1 needs the stored
decoder outputs, figures 3 to 6 need the cached trial table, and every figure
except 6 also reads tables written by `analyses/`, so run those first.

| figure | shows |
|---|---|
| 1 | decoder accuracy, generalisation and attribution |
| 2 | validity of the decoded index |
| 3 | the inverted-U, with both signals at matched scaling |
| 4 | individual differences, from the calibration block |
| 5 | what the feedback conditions do |
| 6 | the optimal trajectory and the deviation metrics |
| 7 | what personalisation can and cannot do |

```bash
python scripts/make_figures.py
```

Pass figure numbers to rebuild a subset, for example `python
scripts/make_figures.py 3 5`. Output goes to `figures/`.

For headless training use the script instead:

```bash
python scripts/train_decoders.py --stage cv --feature-set physio
```

```bash
python scripts/train_decoders.py --stage online
```

`--subjects`, `--repetitions`, `--cpu` and `--suffix` are available for partial
or side-by-side runs.

### Analyses

`analyses/` holds the validation, control and individual-difference analyses.
Each script is standalone, prints a readable report, and writes its tables to
`analyses/results/`. Run them all in dependency order with:

```bash
bash analyses/run_all.sh
```

Four of those scripts retrain decoders and take about five hours between them.
Their per-subject accuracies are committed, so if you only want to reproduce
the statistics that read those outputs, skip the training with:

```bash
bash analyses/run_fast.sh
```

That takes about ten minutes and covers the other twenty-six scripts.

You can also run any one on its own once the trial table exists:

```bash
python analyses/matched_phase.py
```

**Decoder validity**

| script | question |
|---|---|
| `negative_controls` | do shuffled labels, a motor-only decoder or shifted labels reach the same accuracy? |
| `label_controls` | do label controls change enough labels to be informative, and is the label a function of trial phase? |
| `loso_decoder` | what is the accuracy when no data from the test subject is available? |
| `motor_decoder` | does a joystick and head-motion decoder also recover the inverted-U? |
| `clock_check` | is the index a trial clock, and does the motor index track autonomic physiology? |
| `matched_phase` | does the index separate the two courses at matched elapsed time? |
| `scaling_parity` | is the peripheral-over-EEG result an artefact of how the two signals are scaled? |
| `external_validation` | how does the index correlate with raw HR, HRV, EDA and pupil? |

**Arousal and performance**

| script | question |
|---|---|
| `duration_permutation` | what is the p-value once trial-duration structure is preserved? |
| `trial_mean_simulation` | can the trial-mean estimator manufacture an inverted-U? |
| `landmark_analysis` | does the inverted-U hold in a length-safe landmark regression? |
| `hazard_model` | does arousal predict crashing within the next few seconds? |
| `band_limbs` | are both limbs of the inverted-U supported? |
| `failure_locked_arousal` | does arousal escalate into failure events, and does feedback change that? |
| `cluster_permutation` | do the condition trajectories differ, and what size of difference could be detected? |

**Individual differences and control**

| script | question |
|---|---|
| `gamma_definitions` | does the gamma moderation survive a genuine power definition? |
| `baseline_marker_screen` | which baseline measures moderate the arousal-performance curve? |
| `vagal_composite` | does a vagal composite replace the gamma term? |
| `optimum_reliability` | is there a stable per-subject optimum or band width to personalise on? |
| `calibration_profile` | can personalisation work from the calibration block alone? |
| `loso_control_bands` | do the personalised bands survive leave-one-subject-out? |
| `heldout_full_bci` | do they transfer to the full-BCI condition? |
| `band_width_limits` | why does band-width personalisation not improve control? |
| `within_trial_control` | which control parameters add information over the fixed band? |
| `individual_reactivity` | do subjects differ in how much excess arousal costs them, and does a band derived from their own fitted curve beat a universal one? |
| `band_scheme_comparison` | every band scheme scored on the same held-out trials, against a retuned universal band |

**Statistics**

| script | question |
|---|---|
| `clustered_statistics` | what do the trial-level associations look like with subject clustering modelled? |
| `pipeline_sensitivity` | how sensitive are the results to half-sham trials, the gamma definition and the warm-up? |
| `filter_sensitivity` | does any conclusion depend on the smoothing filter? |
| `multiplicity` | one correction policy, applied to every reported p-value |

`multiplicity` transcribes the other scripts' saved outputs, so run it last.

## Package layout

```
arousal/
  config.py    channel maps, condition codes, paths, constants
  extract.py   raw .mat recordings -> ring-locked epochs
  signals.py   despiking, band filtering, normalisation
  models.py    the multimodal decoder
  training.py  cross-validation, attribution, continuous arousal inference
  data.py      epoch loading, feature selection, the trial-level table
  yerkes.py    quadratic mixed-effects arousal-performance models
  control.py   optimal trajectory, deviation metrics, control bands
  plotting.py  shared figure style
analyses/      standalone analysis scripts, one question each
notebooks/     the pipeline end to end, in dependency order
scripts/       headless decoder training and figure generation
```

## Method in brief

**Decoder.** Seven peripheral channels (HR, HRV-pNN35, respiration, phasic and
tonic EDA, left and right pupil) are grouped into four modalities. Each is
encoded by a two-layer 1D convolutional stack with global average pooling into
a 64-dimensional embedding. The four embeddings are concatenated and classified
by a two-layer MLP. Labels follow Faller et al.: large rings are low task
demand, medium and small rings are high demand.

**Continuous arousal.** The decoder is trained on the open-loop calibration
block and applied every 16 samples (62.5 ms) to the most recent 512 samples of
each closed-loop trial. Probabilities are rescaled to 0-100 against the 5th and
95th percentiles of that subject's training distribution, clipped, and smoothed
with an asymmetric IIR filter (rise 0.75 s, fall 1.5 s).

**Validation.** Performance is regressed on arousal and arousal squared with a
random intercept per subject. A negative quadratic coefficient is the
inverted-U. Because trial-mean arousal is averaged over a window whose length is
the outcome, the reported p-value comes from a permutation test that preserves
duration structure. Two length-safe confirmations are provided, a landmark
regression and a within-trial hazard model.

**Control band.** A time-varying optimal arousal trajectory is fitted from the
hard-course control trials. Per-trial deviation from it is related to flight
time, overall and within baseline-physiology subgroups, with
leave-one-subject-out and held-out-condition versions of the same comparison.

`METHODS.md` records the pipeline conventions that are load-bearing but not
obvious from the code, together with reproducibility expectations and runtimes.

## Citation

If you use this code, please cite the software (see `CITATION.cff`) and the
dataset it analyses:

> Faller J, Cummings J, Saproo S, Sajda P (2019). Regulation of arousal via
> online neurofeedback improves human performance in a demanding sensory-motor
> task. *PNAS* 116(13):6482-6490.

## License

MIT. See `LICENSE`.
