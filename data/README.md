# data/

Empty on purpose. No recordings or derived artefacts are distributed with this
repository.

Download the raw `.mat` recordings from
[IEEE DataPort](https://doi.org/10.21227/rn3e-bp31), point `RAW_DIR` in
`notebooks/00_extract_epochs.ipynb` at the folder that contains them, and run
notebooks 00, 01 and 03 in order. They will populate this folder with:

| file | written by | contents |
|---|---|---|
| `ring_events.pkl` | notebook 00 | fixed 2 s epochs ending at each ring crossing |
| `ring_events_online.pkl` | notebook 00 | variable-length epochs that tile each trial |
| `trial_events.pkl` | notebook 00 | per-trial ring-crossing times |
| `results/results_offline_simple_physio.pkl` | notebook 01 | within-subject cross-validation, peripheral |
| `results/results_offline_simple_eeg.pkl` | notebook 01 | within-subject cross-validation, EEG |
| `results/results_offline_simple_all.pkl` | notebook 01 | within-subject cross-validation, combined |
| `results/results_online_simple_physio.pkl` | notebook 01 | continuous decoded arousal per trial |
| `trial_table.pkl` | notebook 03 | the trial-level table every analysis script reads |

The two epoch files are around 4 GB each.
