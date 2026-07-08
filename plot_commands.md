# Plot Commands

Python interpreter: `~/.conda/envs/tagger/bin/python3`

All commands are run from the repo root (`CMSHGCaloChallenge/`).

---

## Summary plots

One command per dataset. Output goes to `plots/summary/{Dataset}/`.

```bash
python3 plot_summary.py --config configs/summary_plot_config_Photon_E5.json
python3 plot_summary.py --config configs/summary_plot_config_Photon_E50.json
python3 plot_summary.py --config configs/summary_plot_config_Photon_E500.json
python3 plot_summary.py --config configs/summary_plot_config_Photon_LogUniform.json
python3 plot_summary.py --config configs/summary_plot_config_Pion_E5.json
python3 plot_summary.py --config configs/summary_plot_config_Pion_E50.json
python3 plot_summary.py --config configs/summary_plot_config_Pion_E500.json
python3 plot_summary.py --config configs/summary_plot_config_Pion_LogUniform.json
```

---

## KS scaling plots

Regenerates all 42 plots (6 categories × 7 datasets) into `eval_scaling/plots/`.
Passes training file lists automatically so the training-shower vertical line appears.
Amplification factors (×N) shown in the legend alongside equivalent shower counts.

```bash
python3 run_scaling_study.py --plot_only --logy
```

---

## 3D event displays

Output root: `plots/event_displays/`

### 500 GeV — all models, single shower + 1k average

```bash
# Photon
python3 plot_3d_event_displays.py \
    --particle Photon --target_energy 500 --energy_label 500 \
    --n_events 1 --n_avg 1000 \
    --output_dir plots/event_displays/

# Pion
python3 plot_3d_event_displays.py \
    --particle Pion --target_energy 500 --energy_label 500 \
    --n_events 1 --n_avg 1000 \
    --output_dir plots/event_displays/
```

Output files (per particle):
- `{particle}_3d_500GeV_all_models_E500.png/.pdf` — multi-panel single shower
- `{particle}_3d_500GeV_all_models_E500_avg1000.png/.pdf` — multi-panel 1k average
- `{Particle}_500GeV/{particle}_{model}.png/.pdf` — individual panels (overwritten by avg run)

### 500 GeV — CaloDiffusion vs Geant4 only

```bash
# Photon
python3 plot_3d_event_displays.py \
    --particle Photon --target_energy 500 --energy_label 500 \
    --n_events 1 --n_avg 1000 \
    --models HGCaloDiffusion \
    --output_dir plots/event_displays/calodiffusion_only/

# Pion
python3 plot_3d_event_displays.py \
    --particle Pion --target_energy 500 --energy_label 500 \
    --n_events 1 --n_avg 1000 \
    --models HGCaloDiffusion \
    --output_dir plots/event_displays/calodiffusion_only/
```

### 50 GeV photon — CaloDiffusion vs Geant4, single shower

```bash
python3 plot_3d_event_displays.py \
    --particle Photon --target_energy 50 --energy_label 50 \
    --n_events 1 \
    --models HGCaloDiffusion \
    --output_dir plots/event_displays/
```

---

## Computational profile plots

Timing bar charts, FLOPs/parameter scatter, and first-batch vs rest comparison.
Output goes to `profiling_plots/`.

```bash
python3 plot_profile.py                      # latency mode (default)
python3 plot_profile.py --throughput         # throughput mode
python3 plot_profile.py --remove_first_batch # exclude warmup batch
```

---

## Pareto plots (quality vs generation time)

Scatter plots of shower quality (AUC − 0.5 or FPD) vs per-shower generation time,
for batch size 1 and batch size 100. Output goes to `profiling_plots/`.

```bash
python3 plot_pareto.py
```

Optional flags:
- `--remove_first_batch` — exclude warmup batch from timing averages
- `--output_dir DIR`     — override output directory (default: `profiling_plots/`)

Output files: `pareto_{auc|fpd}_{photon|pion}_{N}GeV_batch{1|100}.pdf`
