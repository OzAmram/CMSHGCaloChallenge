#!/usr/bin/env python3
"""3D event display: Geant4 vs all generative models for one HGCal shower.

Renders one figure per particle showing Geant4 alongside every available model
on a shared log-energy color scale. Each panel uses cells in (x, y, layer)
space; energy is encoded by color and marker size. Showers are picked from the
variable-energy (1To1000) datasets at a common target incident energy.

Usage:
    python plot_3d_event_displays.py                        # default: Pion + Photon, target 200 GeV
    python plot_3d_event_displays.py --particle Pion --target_energy 100
"""
import argparse
import json
import os

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

import utils
from plotting.plotting_utils import apply_plot_style

try:
    import mplhep as hep
except ImportError:
    hep = None


PHOTON_CONFIG = "config_HGCal_photons.json"
PION_CONFIG = "config_HGCal_pions.json"
DATA_BASE = "/eos/cms/store/group/offcomp-sim/HGCal_Sim_Samples_2024"

# Per-particle list of (display_name, file_list_path) pairs. The first .h5 in
# each list is used. Keep the ordering aligned with the summary plot configs.
MODEL_REGISTRY = {
    "Photon": [
        ("HGCaloDiffusion", "calodiffusion_photon_1To1000_files.txt"),
        ("CaloDream",       "calodream_photon_1To1000_files.txt"),
        ("HGCaloTrilogy",   "hgcalotrilogy_photon_1To1000_files.txt"),
        ("AllShowers",      "thorsten_photon_1To1000_files.txt"),
        ("GraphCNF",        "graphcnf_photon_1To1000_files.txt"),
    ],
    "Pion": [
        ("HGCaloDiffusion", "calodiffusion_pion_1To1000_files.txt"),
        ("CaloDream",       "calodream_pion_1To1000_files.txt"),
        ("HGCaloTrilogy",   "hgcalotrilogy_pion_1To1000_files.txt"),
        ("AllShowers",      "thorsten_pion_1To1000_files.txt"),
    ],
}


def _geant_dir(particle, energy_label):
    return os.path.join(
        DATA_BASE,
        f"Single{particle}_E-{energy_label}_Eta-2_Phi-1p57_Z-321-CloseByParticleGun",
        "Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree",
        "h5s",
    )


def _first_path_in_list(list_path):
    if not os.path.isfile(list_path):
        return None
    with open(list_path) as h:
        for line in h:
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    return None


def _pick_events_by_energy(h5_path, max_cells, shape, target_e, n_events=1,
                           e_tol_rel=0.10):
    """Return up to ``n_events`` showers closest to ``target_e`` in incident energy.

    Yields a list of (shower (nLayers, nCells), picked_E, event_index) tuples,
    sorted by abs distance to target_e (closest first). The h5 file is opened
    only once.
    """
    with h5py.File(h5_path, "r") as h5f:
        if "gen_info" not in h5f:
            raise KeyError(f"{h5_path} has no 'gen_info'")
        gi = np.asarray(h5f["gen_info"][:]).reshape(h5f["gen_info"].shape[0], -1)
        incident_E = gi[:, 0]
        order = np.argsort(np.abs(incident_E - target_e))
        n = min(n_events, order.size)
        picks = sorted(order[:n].tolist())  # sorted for h5py fancy-indexing
        idx_to_rank = {int(idx): rank for rank, idx in enumerate(order[:n])}
        showers_ds = h5f["showers"]
        n_cells_in = showers_ds.shape[2]
        c = min(max_cells, n_cells_in)
        # h5py supports fancy indexing on the first dim if sorted
        raw = showers_ds[picks, :, :c]
        results = [None] * n
        for slot, idx in enumerate(picks):
            out = np.zeros((shape[1], max_cells), dtype=np.float32)
            out[:, :raw.shape[2]] = raw[slot]
            picked_E = float(incident_E[idx])
            rel = abs(picked_E - target_e) / max(target_e, 1.0)
            if rel > e_tol_rel:
                print(f"  WARN {os.path.basename(h5_path)} idx {idx} = "
                      f"{picked_E:.2f} GeV (target {target_e:.1f}, rel {rel:.3f})")
            results[idx_to_rank[int(idx)]] = (out, picked_E, int(idx))
    return results


def _pick_event_by_energy(h5_path, max_cells, shape, target_e, e_tol_rel=0.05):
    """Backwards-compatible single-event picker."""
    res = _pick_events_by_energy(h5_path, max_cells, shape, target_e,
                                 n_events=1, e_tol_rel=e_tol_rel)
    return res[0]


def _shared_norm(shower_arrays, e_floor=1e-3):
    pos = []
    for s in shower_arrays:
        v = s[s > 0]
        if v.size > 0:
            pos.append(v)
    if len(pos) == 0:
        return mcolors.Normalize(vmin=0, vmax=1)
    all_pos = np.concatenate(pos)
    vmin = max(np.quantile(all_pos, 0.05), e_floor)
    vmax = float(all_pos.max())
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)


def _scatter_shower(ax, shower, geom, norm, layer_spacing=2.0, e_min=1e-4,
                    cmap_name="inferno"):
    n_layers, n_cells = shower.shape
    n_cells = min(n_cells, geom.xmap.shape[1])
    xs, ys, zs, es = [], [], [], []
    for ilay in range(n_layers):
        e_lay = shower[ilay, :n_cells]
        active = np.where(e_lay > e_min)[0]
        if active.size == 0:
            continue
        xs.append(geom.xmap[ilay, active])
        ys.append(geom.ymap[ilay, active])
        zs.append(np.full(active.shape, ilay * layer_spacing, dtype=np.float32))
        es.append(e_lay[active])
    if len(xs) == 0:
        return None
    xs, ys, zs, es = map(np.concatenate, (xs, ys, zs, es))

    log_e = np.log10(np.maximum(es, norm.vmin))
    log_lo, log_hi = np.log10(norm.vmin), np.log10(norm.vmax)
    sizes = 3.0 + 50.0 * np.clip((log_e - log_lo) / max(log_hi - log_lo, 1e-9), 0.0, 1.0)

    return ax.scatter(xs, ys, zs, c=es, cmap=cmap_name, norm=norm,
                      s=sizes, alpha=0.85, edgecolors="none")


def _format_axes(ax, n_layers, layer_spacing=2.0, xy_limit=None):
    ax.set_xlabel("x [cm]", labelpad=2, fontsize=9)
    ax.set_ylabel("y [cm]", labelpad=2, fontsize=9)
    ax.set_zlabel("Layer", labelpad=2, fontsize=9)
    n_ticks = 5
    step_layers = max(1, n_layers // n_ticks)
    z_ticks = np.arange(0, n_layers + 1, step_layers) * layer_spacing
    ax.set_zticks(z_ticks)
    ax.set_zticklabels([str(int(z / layer_spacing)) for z in z_ticks])
    ax.tick_params(axis="both", labelsize=8)
    if xy_limit is not None:
        ax.set_xlim(-xy_limit, xy_limit)
        ax.set_ylim(-xy_limit, xy_limit)
    ax.view_init(elev=18, azim=-65)


def _load_layer_weights(layer_weights_file, layer_weights_key, n_layers):
    with open(layer_weights_file) as h:
        layer_weights = np.asarray(json.load(h)[layer_weights_key], dtype=np.float32)
    if layer_weights.size == n_layers + 1:
        layer_weights = layer_weights[1:]
    elif layer_weights.size != n_layers:
        raise ValueError(
            f"layer weights length {layer_weights.size} incompatible with {n_layers}"
        )
    return layer_weights.reshape((-1, 1))


def make_multi_model_event_display(
    particle, output_dir, target_energy=200.0,
    energy_label="1To1000", n_events=1,
    layer_weights_file="HGCalRecHit_layer_weights.json",
    layer_weights_key="weightsPerLayer_V16",
):
    """Render ``n_events`` figures for one (particle, target_energy).

    For event slot i (0..n_events-1): each panel uses the i-th-closest event to
    ``target_energy`` from its own source (Geant4 + each model), so different
    figures show different shower realizations near the same nominal energy.
    """
    apply_plot_style()
    config_file = PHOTON_CONFIG if particle == "Photon" else PION_CONFIG
    with open(config_file) as h:
        cfg = json.load(h)
    geom = utils.load_geom(cfg["BIN_FILE"])
    shape = cfg["SHAPE_ORIG"]
    max_cells = cfg.get("MAX_CELLS", shape[2])
    n_layers = shape[1]
    weights_b = _load_layer_weights(layer_weights_file, layer_weights_key, n_layers)

    # Pick n_events from each source
    g_dir = _geant_dir(particle, energy_label)
    g_files = sorted(f for f in os.listdir(g_dir)
                     if f.startswith("HGCal_showers") and f.endswith(".h5"))
    if not g_files:
        raise FileNotFoundError(f"No Geant4 h5 in {g_dir}")
    geant_path = os.path.join(g_dir, g_files[0])
    geant_picks = _pick_events_by_energy(geant_path, max_cells, shape,
                                         target_energy, n_events=n_events)
    print(f"Geant4: target {target_energy:.0f} GeV -> "
          f"{len(geant_picks)} events ({g_files[0]})")

    model_picks = []
    for display_name, list_name in MODEL_REGISTRY[particle]:
        list_path = os.path.join(os.path.dirname(__file__), list_name)
        h5_path = _first_path_in_list(list_path)
        if not h5_path or not os.path.isfile(h5_path):
            print(f"  skip {display_name}: file list missing or empty ({list_name})")
            continue
        try:
            picks = _pick_events_by_energy(h5_path, max_cells, shape,
                                           target_energy, n_events=n_events)
        except Exception as e:
            print(f"  skip {display_name}: {e}")
            continue
        print(f"{display_name}: {len(picks)} events ({os.path.basename(h5_path)})")
        model_picks.append((display_name, picks))

    finite_xy = np.concatenate([geom.xmap.ravel(), geom.ymap.ravel()])
    finite_xy = finite_xy[np.isfinite(finite_xy)]
    xy_limit = float(np.percentile(np.abs(finite_xy[finite_xy != 0]), 95))

    os.makedirs(output_dir, exist_ok=True)
    # Per-target subfolder for individual single-panel plots.
    indiv_dir = os.path.join(output_dir, f"{particle}_{int(target_energy)}GeV")
    os.makedirs(indiv_dir, exist_ok=True)

    saved_paths = []
    for slot in range(n_events):
        if slot >= len(geant_picks):
            break
        g_shower, g_E, _ = geant_picks[slot]
        panels = [("Geant4", g_shower * weights_b, g_E)]
        for display_name, picks in model_picks:
            if slot >= len(picks):
                continue
            sh, ev_E, _ = picks[slot]
            panels.append((display_name, sh * weights_b, ev_E))

        # Shared color normalization so multi-panel and individual plots agree.
        norm = _shared_norm([p[1] for p in panels], e_floor=1e-3)
        slot_tag = "" if n_events == 1 else f"_s{slot}"

        # 1) Multi-panel summary figure at the top level
        multi_base = _save_multi_panel(
            panels, geom, norm, particle, target_energy, slot, n_events,
            n_layers, xy_limit, output_dir,
            f"{particle.lower()}_3d_{energy_label}GeV_all_models"
            f"_E{int(target_energy)}{slot_tag}",
        )
        saved_paths.append(multi_base)

        # 2) Individual single-panel figures into the per-target subfolder
        for source_label, shower, ev_E in panels:
            indiv_base = _save_single_panel(
                source_label, shower, ev_E, geom, norm,
                particle, target_energy, slot, n_events, n_layers,
                xy_limit, indiv_dir,
                f"{particle.lower()}_{_safe_label(source_label)}{slot_tag}",
            )
            saved_paths.append(indiv_base)
    return saved_paths


def _save_multi_panel(panels, geom, norm, particle, target_energy, slot,
                     n_events, n_layers, xy_limit, output_dir, basename):
    """Multi-panel comparison figure with CMS label header."""
    n = len(panels)
    if n <= 3:
        rows, cols = 1, n
    elif n <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = 2, 3

    fig = plt.figure(figsize=(6.0 * cols, 5.5 * rows + 0.7))
    fig.subplots_adjust(left=0.03, right=0.92, top=0.82, bottom=0.04,
                        wspace=0.05, hspace=0.10)

    ax_label = fig.add_axes([0.04, 0.91, 0.84, 0.06])
    ax_label.set_xticks([])
    ax_label.set_yticks([])
    for spine in ax_label.spines.values():
        spine.set_visible(False)
    if hep is not None:
        hep.cms.label(ax=ax_label, label="Preliminary",
                      data=False, rlabel="Phase-II", fontsize=22)

    sample_tag = "" if n_events == 1 else f", sample {slot + 1}/{n_events}"
    fig.text(0.5, 0.87,
             f"{particle}, target ~{target_energy:.0f} GeV "
             f"(variable-E dataset{sample_tag})",
             fontsize=14, ha="center", color="#333333")

    last_sc = None
    axes_3d = []
    for i, (label, shower, ev_E) in enumerate(panels):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        sc = _scatter_shower(ax, shower, geom, norm)
        last_sc = sc if sc is not None else last_sc
        _format_axes(ax, n_layers, xy_limit=xy_limit)
        ev_str = f"{ev_E:.1f}" if ev_E is not None else "?"
        ax.set_title(f"{label} — {ev_str} GeV", fontsize=13, pad=4)
        axes_3d.append(ax)

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes_3d, shrink=0.65, pad=0.02,
                            location="right")
        cbar.set_label("Cell energy [MeV]", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    base = os.path.join(output_dir, basename)
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(base + ".pdf", dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {base}.png/.pdf")
    return base


def _safe_label(name):
    """Filename-safe version of a model display name."""
    return name.replace(" ", "_").replace("/", "-")


def _save_single_panel(source_label, shower, ev_E, geom, norm,
                       particle, target_energy, slot, n_events,
                       n_layers, xy_limit, output_dir, basename):
    """Render and save one (source × shower) 3D event display."""
    fig = plt.figure(figsize=(8.5, 8.0))
    # Reserve top region for CMS label by leaving a 2D axes pinned at the top.
    fig.subplots_adjust(left=0.05, right=0.88, top=0.86, bottom=0.04)

    # Top label band: hidden 2D axes hosting hep.cms.label so the CMS / qualifier
    # text uses the same font and weights as the summary plots.
    ax_label = fig.add_axes([0.05, 0.88, 0.83, 0.08])
    ax_label.set_xticks([])
    ax_label.set_yticks([])
    for spine in ax_label.spines.values():
        spine.set_visible(False)
    if hep is not None:
        hep.cms.label(ax=ax_label, label="Preliminary",
                      data=False, rlabel="Phase-II", fontsize=22)
    else:  # pragma: no cover — mplhep should always be installed
        ax_label.text(0.0, 0.5, "CMS", fontsize=22, fontweight="bold",
                      transform=ax_label.transAxes, va="center")
        ax_label.text(0.13, 0.5, "Simulation Preliminary",
                      fontsize=18, style="italic",
                      transform=ax_label.transAxes, va="center")
        ax_label.text(1.0, 0.5, "Phase-II", fontsize=16,
                      transform=ax_label.transAxes, va="center", ha="right")

    ax = fig.add_subplot(111, projection="3d")
    sc = _scatter_shower(ax, shower, geom, norm)
    _format_axes(ax, n_layers, xy_limit=xy_limit)
    # Increase tick fontsize relative to the multi-panel layout
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    ax.zaxis.label.set_size(13)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.08, location="right")
        cbar.set_label("Cell energy [MeV]", fontsize=13)
        cbar.ax.tick_params(labelsize=11)

    ev_str = f"{ev_E:.1f}" if ev_E is not None else "?"
    sample_tag = "" if n_events == 1 else f", sample {slot + 1}/{n_events}"
    ax.set_title(
        f"{source_label} — {particle}, {ev_str} GeV"
        f"{sample_tag}",
        fontsize=15, pad=8,
    )

    base = os.path.join(output_dir, basename)
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(base + ".pdf", dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {base}.png/.pdf")
    return base


DEFAULT_TARGETS = [
    ("Photon", 5.0),
    ("Photon", 20.0),
    ("Photon", 80.0),
    ("Photon", 200.0),
    ("Photon", 500.0),
    ("Pion", 10.0),
    ("Pion", 30.0),
    ("Pion", 100.0),
    ("Pion", 300.0),
    ("Pion", 700.0),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="plots/event_displays/")
    parser.add_argument("--particle", default=None,
                        choices=["Photon", "Pion", "all"],
                        help="If set, override default 10-target list and "
                             "use a single (particle, target_energy) pair.")
    parser.add_argument("--target_energy", type=float, default=200.0,
                        help="Target incident energy [GeV] (only with --particle)")
    parser.add_argument("--energy_label", default="1To1000",
                        help="Variable-energy label (default 1To1000)")
    parser.add_argument("--targets", nargs="*", default=None,
                        help="Override DEFAULT_TARGETS as 'Particle:E' pairs, "
                             "e.g. Photon:50 Pion:200")
    parser.add_argument("--n_events", type=int, default=3,
                        help="Number of events to render per target energy "
                             "(default 3 — different figures showing the i-th "
                             "closest event to target_E from each source)")
    args = parser.parse_args()

    if args.targets:
        plan = []
        for entry in args.targets:
            p, e = entry.split(":", 1)
            plan.append((p.strip(), float(e)))
    elif args.particle is not None and args.particle != "all":
        plan = [(args.particle, args.target_energy)]
    else:
        plan = DEFAULT_TARGETS

    for particle, target_e in plan:
        make_multi_model_event_display(
            particle, args.output_dir,
            target_energy=target_e,
            energy_label=args.energy_label,
            n_events=args.n_events,
        )


if __name__ == "__main__":
    main()
