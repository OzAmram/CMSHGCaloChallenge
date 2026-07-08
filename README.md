# CMSHGCaloChallenge

Evaluation framework for generative models of the CMS High Granularity Calorimeter (HGCal).

---

## Dataset splits

The challenge uses a 70/30 train/test split. Train and test file lists are in [`datasets/`](./datasets).
The test set must not be used for training, validation, or any optimization.

| Particle | Train/val (`*_train.txt`) | Test (`*_test.txt`) |
|----------|--------------------------|---------------------|
| Photon   | files 0–244              | files 245–350       |
| Pion     | files 0–249              | files 250–360       |

Generated sample file lists (per model per dataset) are in [`datasets/generated/`](./datasets/generated/).

---

## Running evaluations

Evaluation metrics are computed with `hgcal_metrics.py`:

```bash
python hgcal_metrics.py \
    -c configs/config_HGCal_pions.json \
    -g datasets/generated/MyModel_Pion_E50.txt \
    -p eval_results_all/MyModel/Pion_E50/ \
    -d /path/to/geant/pion/h5s/ \
    --mode all --plot --name MyModel
```

Key arguments:
- `-c` — config JSON in `configs/` (`config_HGCal_photons.json`, `config_HGCal_photons_E5.json`, etc.)
- `-g` — text file listing paths to generated shower HDF5 files
- `-p` — output directory (metrics.txt, hist/*.npz, hist/*.png written here)
- `-d` — directory containing Geant4 reference showers
- `--mode` — `hist`, `cls`, `fpd`, or `all` (default: `all`)
- `--plot` — save per-feature histogram plots
- `--EMin` — minimum voxel energy threshold (default: 1e-5)

Feature npz files are cached alongside each input HDF5 (`.feat.npz` suffix); use `--reprocess` to force recomputation.

---

## Summary table

Print a text + LaTeX table of all model metrics across all datasets:

```bash
python print_summary.py
```

Write all metrics to a single JSON file for downstream use:

```bash
python print_summary.py --json metrics_summary.json
```

A pre-generated `metrics_summary.json` is included in the repo.

---

## Histogram overlay plots

Overlay per-feature histograms from multiple models using pre-computed `hist/*.npz` files:

```bash
python plot_summary.py --config configs/summary_plot_config_Photon_E50.json
```

Config files for all 8 datasets are in `configs/summary_plot_config_*.json`.
The hist npz files needed to run these plots are committed under `eval_results_all/*/hist/*.npz`.

---

## KS scaling study

Diagnose each model's fidelity in terms of an equivalent number of Geant4 samples:

```bash
# Replot from committed scaling data (no EOS access needed)
python run_scaling_study.py --plot_only --logy

# Run the full scaling study + plot (requires EOS access)
python run_scaling_study.py --datasets Photon_E5 Pion_E5
```

Pre-computed scaling curves for all 6 single-energy datasets are in
`eval_scaling/ks_scaling_all.json`. Plots go to `eval_scaling/plots/`.

---

## 3D event displays

Render 3D shower visualizations for one or more models:

```bash
# 500 GeV photons, all models
python plot_3d_event_displays.py \
    --particle Photon --target_energy 500 --energy_label 500 \
    --n_events 1 --n_avg 1000 \
    --output_dir plots/event_displays/

# Single model only
python plot_3d_event_displays.py \
    --particle Photon --target_energy 500 --energy_label 500 \
    --models HGCaloDiffusion \
    --n_events 1 --n_avg 1000 \
    --output_dir plots/event_displays/calodiffusion_only/
```

---

## Computational profile plots

Timing bar charts, FLOPs/parameter scatter, and first-batch-vs-rest comparison:

```bash
python3 plot_profile.py                      # latency mode
python3 plot_profile.py --throughput         # throughput mode
python3 plot_profile.py --remove_first_batch # exclude warmup batch
```

Input data lives in `timing_inputs/`; outputs go to `profiling_plots/`.

---

## Pareto plots (quality vs generation time)

Scatter plots of shower quality vs per-shower generation time for comparing the
speed–fidelity tradeoff across models. Quality metrics are AUC − 0.5 (lower =
closer to Geant4) and FPD. Produced for batch size 1 and batch size 100 at each
energy and particle type:

```bash
python3 plot_pareto.py
```

Outputs: `profiling_plots/pareto_{auc|fpd}_{photon|pion}_{N}GeV_batch{1|100}.pdf`

---

## Reproducing all plots

See [`plot_commands.md`](./plot_commands.md) for the exact commands used to regenerate
all summary, scaling, 3D event display, profiling, and Pareto plots.
