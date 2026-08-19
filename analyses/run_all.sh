#!/usr/bin/env bash
# Run every analysis script in dependency order.
#
# Four scripts train decoders and dominate the runtime. On a single GPU,
# budget roughly 1 h for negative_controls and motor_decoder together, 1.5 h for
# loso_decoder (16 folds x 3 repetitions) and 1.5 h for label_controls (6 label
# conditions x 16 subjects). cluster_permutation takes about 5 min for 10,000
# permutations plus the sensitivity simulation. Everything else runs in seconds
# to a few minutes.
#
# loso_decoder resumes from results/loso_decoder.csv if it is interrupted.
# Delete that file to force a clean run.
#
# Dependencies that fix the order below:
#   motor_decoder          writes the motor trial table read by matched_phase,
#                          hazard_model and clock_check
#   baseline_marker_screen provides baseline_panel() to vagal_composite and
#                          calibration_profile
#   label_controls         writes the out-of-fold probabilities that
#                          calibration_profile builds its decoded measures from
#   calibration_profile    provides build_profile() to band_width_limits and
#                          within_trial_control, and writes the calibration
#                          measures that individual_reactivity reads
#   within_trial_control   writes the per-second bins that individual_reactivity
#                          fits its hazard models on
#   individual_reactivity  writes the curve-derived band frames that
#                          band_scheme_comparison scores against the universal
#                          and retuned bands
#   multiplicity           transcribes the other scripts' outputs, so it is last

set -e
cd "$(dirname "$0")"

for script in \
    loso_control_bands heldout_full_bci clustered_statistics \
    failure_locked_arousal pipeline_sensitivity negative_controls \
    motor_decoder gamma_definitions trial_mean_simulation \
    landmark_analysis baseline_marker_screen matched_phase \
    duration_permutation vagal_composite hazard_model band_limbs \
    loso_decoder scaling_parity label_controls external_validation \
    cluster_permutation clock_check optimum_reliability \
    calibration_profile band_width_limits within_trial_control \
    individual_reactivity band_scheme_comparison filter_sensitivity \
    multiplicity
do
    echo "=== $script"
    python3 "$script.py" 2>&1 | tee "results/${script}.log"
done
