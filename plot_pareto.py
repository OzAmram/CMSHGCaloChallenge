#!/usr/bin/env python3
"""Pareto plots: shower quality vs generation time per model."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import mplhep as hep
from pathlib import Path

plt.style.use(hep.style.CMS)
mpl.rcParams.update({
    "axes.labelpad": 5,
    "legend.frameon": False,
    "legend.handletextpad": 0.8,
    "yaxis.labellocation": "center",
})

MODEL_COLORS = {
    "HGCaloDiffusion": "#5790fc",
    "HGCaloDream":     "#f89c20",
    "HGCaloTrilogy":   "#e42536",
    "GraphCNF":        "#964a8b",
    "AllShowers":      "#9c9ca1",
    "CaloDiT-2":       "#7a21dd",
}


def _parse_name(raw: str) -> tuple[str, str]:
    """Return (model_display_name, particle) from a timing entry name."""
    particle = "Photon" if "photon" in raw.lower() else "Pion"
    model = raw.replace("Photon", "").replace("Pion", "")
    return model, particle


def collect_timing(data: list, batch_size: int, remove_first_batch: bool = False) -> list[dict]:
    """Compute mean per-shower latency (s/shower) for a given batch size."""
    out = []
    for pt in data:
        if pt["batch"] != batch_size:
            continue
        if remove_first_batch:
            times = np.array([v for k, v in pt["timing"].items()
                              if "sample_batch" in k and k != "sample_batch_0"])
        else:
            times = np.array([v for k, v in pt["timing"].items() if "sample_batch" in k])

        model, particle = _parse_name(pt["name"])
        out.append({
            "model":     model,
            "particle":  particle,
            "energy":    pt["energy"],
            "time_mean": float(np.mean(times / pt["samples"])),
            "time_std":  float(np.std(times  / pt["samples"])),
        })
    return out


def _extend_axes_for_legend(ax, fig, xs, ys):
    """Extend the x-axis (log scale) rightward so the upper-right legend
    does not overlap any plotted (x, y) points."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = ax.get_legend()
    if legend is None:
        return

    # Legend bbox in axes-fraction coords
    leg_disp = legend.get_window_extent(renderer)
    ax_disp  = ax.get_window_extent(renderer)
    leg_x0_frac = (leg_disp.x0 - ax_disp.x0) / ax_disp.width
    leg_y0_frac = (leg_disp.y0 - ax_disp.y0) / ax_disp.height

    # Current log x limits
    log_xmin, log_xmax = np.log10(ax.get_xlim()[0]), np.log10(ax.get_xlim()[1])
    ymin, ymax = ax.get_ylim()

    def frac_to_logx(f):
        return 10 ** (log_xmin + f * (log_xmax - log_xmin))

    if ax.get_yscale() == "log":
        log_ymin, log_ymax = np.log10(ymin), np.log10(ymax)
        def frac_to_y(f):
            return 10 ** (log_ymin + f * (log_ymax - log_ymin))
    else:
        def frac_to_y(f):
            return ymin + f * (ymax - ymin)

    leg_x0_data = frac_to_logx(leg_x0_frac)
    leg_y0_data = frac_to_y(leg_y0_frac)

    # Points that fall inside the legend region (need clearing)
    conflict_xs = [x for x, y in zip(xs, ys) if x >= leg_x0_data and y >= leg_y0_data]
    if not conflict_xs:
        return

    # We need max(conflict_xs) to map to at most leg_x0_frac in axes coords,
    # with a 5 % padding so the point sits just outside the legend.
    max_cx = max(conflict_xs)
    target_frac = leg_x0_frac * 0.95          # leave a small gap
    # target_frac = (log10(max_cx) - log_xmin) / new_log_range
    new_log_range = (np.log10(max_cx) - log_xmin) / target_frac
    ax.set_xlim(10 ** log_xmin, 10 ** (log_xmin + new_log_range))


def plot_pareto(timing_data: list, metrics: dict,
                particle: str, energy: int,
                metric: str, batch_size: int,
                out_path: Path) -> None:
    """
    Scatter plot of quality metric vs generation time for all models.

    Args:
        timing_data: output of collect_timing()
        metrics:     {display_name -> {dataset -> {metric -> value}}}
        metric:      "AUC" (plotted as |AUC-0.5|) or "FPD"
        batch_size:  1 or 10 (used only for the axis label)
    """
    dataset_key = f"{particle}_E{energy}"

    fig, ax = plt.subplots(figsize=(10, 8))

    plotted = []
    xs, ys = [], []
    for pt in timing_data:
        if pt["particle"].lower() != particle.lower():
            continue
        if pt["energy"] != energy:
            continue
        model = pt["model"]
        if model not in MODEL_COLORS:
            continue

        model_metrics = metrics.get(model, {}).get(dataset_key, {})

        if metric == "AUC":
            raw = model_metrics.get("AUC")
            if raw is None:
                continue
            y_val = abs(raw - 0.5)
            y_err = None
        else:
            y_val = model_metrics.get("FPD")
            y_err = model_metrics.get("FPD_err")
            if y_val is None:
                continue

        color = MODEL_COLORS[model]
        ax.errorbar(
            pt["time_mean"], y_val,
            xerr=pt["time_std"],
            yerr=y_err,
            color=color,
            marker="o", alpha=0.8,
            ms=1, elinewidth=1, capsize=3,
        )
        ax.scatter(
            pt["time_mean"], y_val,
            color=color, marker="o",
            alpha=0.9, s=100,
            edgecolors="black", linewidths=0.5,
        )
        plotted.append(model)
        xs.append(pt["time_mean"])
        ys.append(y_val)

    if not plotted:
        plt.close()
        print(f"No data for {particle} {energy} GeV {metric} batch={batch_size}, skipping.")
        return

    hep.cms.label("Preliminary", ax=ax, data=False,
                  rlabel=f"{particle} {energy} GeV", loc=0)

    ax.set_xscale("log")
    ax.set_xlabel(f"Time/Shower (s) [batch size = {batch_size}]", fontsize=24)

    if metric == "AUC":
        ax.set_ylabel(r"AUC $-$ 0.5", fontsize=24)
    else:
        ax.set_yscale("log")
        ax.set_ylabel(r"FPD ($\times 10^{-3}$)", fontsize=24)

    legend_handles = [
        mpatches.Patch(color=color, label=label, ec="black")
        for label, color in MODEL_COLORS.items()
        if label in plotted
    ]
    ax.legend(handles=legend_handles, fontsize=16, loc="upper right")

    # Extend x-axis so no points sit behind the legend
    plt.tight_layout()
    _extend_axes_for_legend(ax, fig, xs, ys)

    # Arrow indicating direction of improvement (bottom-left = fast AND good quality)
    ax.annotate(
        "Better",
        xy=(0.08, 0.07), xycoords="axes fraction",
        xytext=(0.22, 0.19), textcoords="axes fraction",
        fontsize=17, fontweight="bold", color="dimgray",
        ha="center", va="center",
        arrowprops=dict(
            arrowstyle="-|>",
            color="dimgray",
            lw=3,
            mutation_scale=28,
        ),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing_input",  default="timing_inputs/combined_timing_profiles.json")
    parser.add_argument("--metrics_input", default="metrics_summary.json")
    parser.add_argument("--output_dir",    default="profiling_plots")
    parser.add_argument("--remove_first_batch", action="store_true",
                        help="Exclude first (warmup) batch from timing averages")
    args = parser.parse_args()

    timing_raw = json.loads(Path(args.timing_input).read_text())
    metrics    = json.loads(Path(args.metrics_input).read_text())
    output_dir = Path(args.output_dir)

    particles = ["Photon", "Pion"]
    energies  = [5, 50, 500]

    for batch_size in [1, 100]:
        timing_data = collect_timing(timing_raw, batch_size,
                                     remove_first_batch=args.remove_first_batch)
        for particle in particles:
            for energy in energies:
                for metric in ["AUC", "FPD"]:
                    fname = (f"pareto_{metric.lower()}_{particle.lower()}"
                             f"_{energy}GeV_batch{batch_size}.pdf")
                    plot_pareto(timing_data, metrics, particle, energy,
                                metric, batch_size, output_dir / fname)
