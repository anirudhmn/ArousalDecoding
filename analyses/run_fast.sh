#!/usr/bin/env bash
# Re-run every analysis that does not train a decoder.
#
# The four training scripts (negative_controls, motor_decoder, loso_decoder,
# label_controls) take about 5 h combined and their outputs are already in
# results/. Everything below reads those outputs and recomputes the statistics
# the manuscript reports. Order matches run_all.sh.

cd "$(dirname "$0")"

for script in \
    loso_control_bands heldout_full_bci clustered_statistics \
    failure_locked_arousal pipeline_sensitivity \
    gamma_definitions trial_mean_simulation \
    landmark_analysis baseline_marker_screen matched_phase \
    duration_permutation vagal_composite hazard_model band_limbs \
    scaling_parity external_validation \
    cluster_permutation clock_check optimum_reliability \
    calibration_profile band_width_limits within_trial_control \
    individual_reactivity band_scheme_comparison filter_sensitivity \
    multiplicity
do
    echo "=== $script"
    python3 "$script.py" > "results/${script}.log" 2>&1
    echo "    exit $?"
done
echo "=== done"
