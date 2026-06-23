#!/usr/bin/env python3
"""Plot KS metric vs generated sample size from eval_scaling directories."""

import re
import sys
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO        = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from plotting_utils import apply_plot_style, CMS_COLORS

try:
    import mplhep as hep
except ImportError:
    hep = None

SCALING     = REPO / "eval_scaling"
RESULTS_ALL = REPO / "eval_results_all"
CATS        = ["Energy", "Transverse", "Center", "Width", "Occupancy", "all"]
CAT_LABELS  = {"all": "All"}   # overrides for display/filenames; others use cat as-is

# display name -> directory name(s) in eval_results_all, tried in order;
# first one with a metrics.txt for the requested dataset wins. This lets a
# model's directory be upgraded (e.g. "_v2") without losing datasets that
# haven't been re-run under the new directory yet.
MODELS = {
    "HGCaloDiffusion": ["CaloDiffusion"],
    "CaloDream":       ["CaloDream_v3", "CaloDream"],
    "HGCaloTrilogy":   ["GLAM"],
    "GraphCNF":        ["GraphCNF_v2", "GraphCNF"],
    "AllShowers":      ["Thorsten"],
    "CaloDiT":         ["CaloDiT"],
}
MODEL_COLORS = CMS_COLORS


def parse_metrics(path: Path) -> dict:
    result = {}
    for line in path.read_text().splitlines():
        m = re.match(r"Num generated showers:\s*(\d+)", line)
        if m:
            result["n_showers"] = int(m.group(1))
        m = re.match(r"Avg separation power / KS of (\w+) features:\s*([\d.e+\-]+)\s*/\s*([\d.e+\-]+)", line)
        if m:
            cat, sep, ks = m.group(1), float(m.group(2)), float(m.group(3))
            result[f"KS_{cat}"]  = ks
            result[f"Sep_{cat}"] = sep
    return result


def load_scaling_data(pattern: str) -> list[dict]:
    rows = []
    for d in sorted(SCALING.glob(pattern)):
        m_path = d / "metrics.txt"
        if not m_path.exists():
            continue
        m = parse_metrics(m_path)
        if not m:
            continue
        if "n_showers" not in m:
            num = re.search(r"n(\d+)", d.name)
            if num:
                m["n_showers"] = int(num.group(1))
        rows.append(m)
    rows.sort(key=lambda r: r.get("n_showers", 0))
    return rows


def count_training_showers(file_list: str, directory: str, energy: float, windows=(0.006,)) -> dict:
    """Count showers within each fractional energy window across all h5 files."""
    p = Path(file_list)
    raw = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    h5files = [str(Path(directory) / Path(f).name) for f in raw] if directory else raw

    totals = {w: 0 for w in windows}
    for h5path in h5files:
        try:
            with h5py.File(h5path, "r") as f:
                energies = f["gen_info"][:, 0].squeeze()
            frac_diff = np.abs(energies - energy) / energy
            for w in windows:
                totals[w] += int(np.sum(frac_diff <= w))
        except Exception as e:
            print(f"  WARNING: skipping {h5path}: {e}")
    return totals


def load_model_metric(dataset: str, key: str) -> dict:
    """Return {display_name: metric_value} for each model on the given dataset."""
    result = {}
    for name, dirnames in MODELS.items():
        for dirname in dirnames:
            p = RESULTS_ALL / dirname / dataset / "metrics.txt"
            if p.exists():
                break
        else:
            continue
        m = parse_metrics(p)
        val = m.get(key)
        if val is not None:
            result[name] = val
    return result


def equiv_n(ns: np.ndarray, ks_curve: np.ndarray, model_ks: float) -> tuple:
    """Inverse-interpolate: find n where the Geant4 curve equals model_ks.

    Returns (n_eq, status) where status is one of:
      'ok'        — interpolated normally
      'below_min' — model metric < curve minimum (better than best measured n)
      'above_max' — model metric > curve maximum (worse than worst measured n)
    """
    mask = ~np.isnan(ks_curve)
    ns_clean, ks_clean = ns[mask], ks_curve[mask]
    if len(ks_clean) < 2:
        return np.nan, "ok"
    ns_flip = ns_clean[::-1]
    ks_flip = ks_clean[::-1]
    if model_ks < ks_flip[0]:
        return float(ns_flip[0]), "below_min"
    if model_ks > ks_flip[-1]:
        return float(ns_flip[-1]), "above_max"
    return float(np.interp(model_ks, ks_flip, ns_flip)), "ok"


def _decimate_curve(ns: np.ndarray, ys: np.ndarray, min_log_gap: float = 0.01) -> tuple:
    """Merge curve points whose n_showers are within min_log_gap (log10) of each
    other, averaging their y-values.

    Some n_showers values were re-run independently (e.g. n=10000 appears twice,
    from separate random draws with different sampling noise) and land at nearly
    the same x with different y, which shows up as a sharp zigzag in the line
    plot. Merging close points smooths this out.
    """
    if len(ns) <= 1:
        return ns, ys
    order = np.argsort(ns)
    ns_s, ys_s = ns[order], ys[order]
    log_ns = np.log10(np.maximum(ns_s, 1e-12))

    merged_ns, merged_ys = [], []
    group_logs, group_ys = [log_ns[0]], [ys_s[0]]
    for i in range(1, len(ns_s)):
        if log_ns[i] - group_logs[0] <= min_log_gap:
            group_logs.append(log_ns[i])
            group_ys.append(ys_s[i])
        else:
            merged_ns.append(10 ** np.mean(group_logs))
            merged_ys.append(np.mean(group_ys))
            group_logs, group_ys = [log_ns[i]], [ys_s[i]]
    merged_ns.append(10 ** np.mean(group_logs))
    merged_ys.append(np.mean(group_ys))
    return np.array(merged_ns), np.array(merged_ys)


def _extend_limits_for_legend(ax, fig, vline_xs, curve_ns, curve_ys,
                               x_train_max=None, legend_gap=0.03,
                               logx=False, logy=False, xmax_cap=None):
    """Extend x/y limits so the legend doesn't overlap data or vertical lines.

    If x_train_max is given, the legend's left edge is anchored just to the right
    of that training line (legend_gap in axes fraction), and xmax is extended only
    enough to fit the legend width.  Otherwise the original behaviour applies:
    xmax is extended so the rightmost vline sits left of the fixed right-anchored
    legend.
    """
    fig.canvas.draw()
    legend = ax.get_legend()
    if legend is None:
        return

    renderer = fig.canvas.get_renderer()
    leg_bb = legend.get_window_extent(renderer)
    ax_bb  = ax.get_window_extent(renderer)

    W   = leg_bb.width  / ax_bb.width           # legend width in axes fraction
    lx0 = max((leg_bb.x0 - ax_bb.x0) / ax_bb.width  - 0.02, 0.01)
    ly0 = max((leg_bb.y0 - ax_bb.y0) / ax_bb.height - 0.02, 0.01)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    new_xmax = xmax

    if x_train_max is not None and xmax_cap is None:
        # Extend xmax so the training line sits at fraction (1 - gap - W - margin),
        # leaving exactly enough room for the legend to its right.
        train_frac = max(1.0 - legend_gap - W - 0.01, 0.1)
        if logx:
            log_xmin   = np.log10(max(xmin,        1e-12))
            log_xtrain = np.log10(max(x_train_max, 1e-12))
            new_xmax   = max(new_xmax,
                             10 ** (log_xmin + (log_xtrain - log_xmin) / train_frac))
            x_anchor   = ((log_xtrain - log_xmin)
                          / (np.log10(max(new_xmax, 1e-12)) - log_xmin)
                          + legend_gap)
        else:
            new_xmax = max(new_xmax, xmin + (x_train_max - xmin) / train_frac)
            x_anchor = (x_train_max - xmin) / (new_xmax - xmin) + legend_gap

        # Reposition legend: left edge at x_anchor (upper-left anchor, loc=2)
        legend.set_bbox_to_anchor((x_anchor, 0.86), transform=ax.transAxes)
        legend._loc = 2   # "upper left" — upper-left corner at the anchor point
        lx0_for_y = x_anchor - 0.01

    else:
        # Push the rightmost vline right a bit so it tends to clear the fixed
        # legend, using a fixed modest multiplier rather than solving for
        # guaranteed clearance — dividing by the legend's (sometimes small)
        # width fraction caused the x-axis to blow out to absurd values.
        if vline_xs and xmax_cap is None:
            x_r = max(v for v in vline_xs if np.isfinite(v))
            if logx:
                new_xmax = max(new_xmax, x_r * 10)
            else:
                new_xmax = max(new_xmax, x_r * 1.5)
        lx0_for_y = lx0

    ax.set_xlim(xmin, new_xmax)

    # Legend left edge in data coords for ymax computation
    if logx:
        log_xmin     = np.log10(max(xmin,     1e-12))
        log_new_xmax = np.log10(max(new_xmax, 1e-12))
        leg_x_left   = 10 ** (log_xmin + lx0_for_y * (log_new_xmax - log_xmin))
    else:
        leg_x_left = xmin + lx0_for_y * (new_xmax - xmin)

    # Push ymax up so the Geant4 curve in the legend's x-range sits below it
    new_ymax = ymax
    if curve_ns is not None and len(curve_ns) and ly0 > 0:
        in_leg = curve_ns >= leg_x_left
        if in_leg.any():
            y_top = float(np.nanmax(curve_ys[in_leg]))
            if y_top > ymin:
                if logy:
                    log_ymin = np.log10(max(ymin, 1e-12))
                    new_ymax = max(new_ymax,
                                   10 ** (log_ymin
                                          + (np.log10(max(y_top, 1e-12)) - log_ymin) / ly0))
                else:
                    new_ymax = max(new_ymax, ymin + (y_top - ymin) / ly0)

    ax.set_ylim(ymin, new_ymax)


def _format_dataset(dataset: str) -> str:
    """'Photon_E50' -> 'Photon, 50 GeV'."""
    m = re.match(r"(\w+)_E(\d+)$", dataset)
    if m:
        return f"{m.group(1)}s, {m.group(2)} GeV"
    return dataset.replace("_", " ")


def make_panel(cat, metric_key_prefix, metric_label, flags, rows, ns, n_train):
    """Create a standalone CMS-styled figure for one (metric, category) pair."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    curve      = np.array([r.get(f"{metric_key_prefix}_{cat}", np.nan) for r in rows])
    model_vals = load_model_metric(flags.dataset, f"{metric_key_prefix}_{cat}")

    mask = ~np.isnan(curve)
    curve_ns = ns[mask]
    curve_ys = curve[mask] * 1e3
    curve_ns, curve_ys = _decimate_curve(curve_ns, curve_ys)
    ax.plot(curve_ns, curve_ys, marker="o", color="black",
            linewidth=2, label="Geant4 self-comparison", zorder=3)

    # Floor for model vlines: 0 on linear, small positive value on log
    pos_ys = curve_ys[curve_ys > 0]
    y_floor = float(pos_ys.min()) * 0.3 if (flags.logy and len(pos_ys)) else 0

    vline_xs = []
    hline_ys = []
    x_lo = float(ns[mask].min()) if mask.any() else 1
    n_min_curve = x_lo
    n_max_curve = float(ns[mask].max()) if mask.any() else 1

    offchart_handles = []   # (proxy_handle, label) for models off the measured curve
    for (name, val), color in zip(model_vals.items(), MODEL_COLORS):
        n_eq, status = equiv_n(ns, curve, val)
        y_val = val * 1e3

        if status == "ok":
            hline_ys.append(y_val)
            ax.vlines(n_eq, ymin=y_floor, ymax=y_val, color=color, linestyle="--",
                      linewidth=1.5, label=f"{name} ({n_eq:,.0f})")
            ax.hlines(y_val, xmin=x_lo, xmax=n_eq, color=color, linestyle="--",
                      linewidth=1.5)
            vline_xs.append(n_eq)
        elif status == "below_min":
            # Model outperforms the entire measured curve — legend-only, no line drawn
            # (avoids dragging the y-range out to fit an off-chart outlier)
            print(f"  [{cat}] {name}: {metric_label}={val:.3e} — better than curve (>{n_max_curve:,.0f})")
            offchart_handles.append((plt.Line2D([], [], color=color, linestyle="--", linewidth=1.5),
                                      f"{name} (>{n_max_curve:,.0f})"))
        else:  # above_max
            # Model underperforms even the smallest sample size — legend-only, no line drawn
            print(f"  [{cat}] {name}: {metric_label}={val:.3e} — worse than curve (<{n_min_curve:,.0f})")
            offchart_handles.append((plt.Line2D([], [], color=color, linestyle="--", linewidth=1.5),
                                      f"{name} (<{n_min_curve:,.0f})"))

    _line_styles = ["-", "--", ":", "-."]
    for i, (w, n) in enumerate(n_train.items()):
        ls = _line_styles[i % len(_line_styles)]
        ax.axvline(n, color="grey", linestyle=ls, linewidth=2,
                   label=f"Training E±{w*100:.4g}% ({n:,})", zorder=4)
        vline_xs.append(n)

    ax.set_xlabel("Number of showers")
    ax.set_ylabel(rf"{metric_label} ($\times 10^{{-3}}$)")

    if flags.logy:
        ax.set_yscale("log")
        y_lo = y_floor if y_floor > 0 else (pos_ys.min() * 0.3 if len(pos_ys) else 1e-3)
        ax.set_ylim(bottom=y_lo)
    elif hline_ys:
        ax.set_ylim(bottom=0, top=max(hline_ys) * 1.15)
    else:
        ax.set_ylim(bottom=0)

    if flags.logx:
        ax.set_xscale("log")

    if vline_xs:
        x_lo_v = min(vline_xs)
        x_hi_v = max(vline_xs) if flags.xmax is None else flags.xmax
        if flags.logx:
            log_lo = np.log10(max(x_lo_v, 1))
            log_hi = np.log10(max(x_hi_v, 1))
            pad = 0.1 * (log_hi - log_lo)
            ax.set_xlim(left=10 ** (log_lo - pad), right=10 ** (log_hi + pad))
        else:
            pad = 0.05 * (x_hi_v - x_lo_v)
            ax.set_xlim(left=max(0, x_lo_v - pad), right=x_hi_v + pad)

    ax.grid(True, alpha=0.3)

    # Dataset + category label inside the frame, top-right, above the legend
    dataset_str = _format_dataset(flags.dataset)
    cat_label = CAT_LABELS.get(cat, cat)
    ax.text(0.97, 0.92, f"{dataset_str}: {cat_label} Features",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=14, fontweight="bold")

    # Two-column legend: left col = Geant4 + training, right col = models.
    # matplotlib fills columns top-to-bottom left-first, so concatenate the two
    # groups (ref padded to max_len, then models padded to max_len) rather than
    # interleaving, so the first half lands in col 0 and the second in col 1.
    handles, labels = ax.get_legend_handles_labels()
    ref_hl   = [(h, l) for h, l in zip(handles, labels)
                if "Geant4" in l or "Training" in l]
    model_hl = [(h, l) for h, l in zip(handles, labels)
                if "Geant4" not in l and "Training" not in l]
    model_hl += offchart_handles

    blank = (plt.Line2D([], [], linewidth=0, alpha=0), "")
    leg_kw = dict(fontsize=10, loc="upper right", bbox_to_anchor=(0.99, 0.86),
                  handlelength=1.5, columnspacing=0.8, handletextpad=0.5,
                  frameon=False)
    if ref_hl and model_hl:
        max_len  = max(len(ref_hl), len(model_hl))
        ref_hl  += [blank] * (max_len - len(ref_hl))
        model_hl += [blank] * (max_len - len(model_hl))
        all_h = [h for h, _ in ref_hl] + [h for h, _ in model_hl]
        all_l = [l for _, l in ref_hl] + [l for _, l in model_hl]
        ax.legend(all_h, all_l, ncols=2, **leg_kw)
    else:
        ax.legend(handles, labels, **leg_kw)

    # Extend x/y limits so no lines or curve overlap the legend
    x_train_max = max(n_train.values()) if n_train else None
    _extend_limits_for_legend(
        ax, fig, vline_xs, curve_ns, curve_ys,
        x_train_max=x_train_max,
        logx=flags.logx, logy=flags.logy, xmax_cap=flags.xmax,
    )

    # CMS label with reduced font size; dataset info is already inside the frame
    if hep is not None:
        try:
            hep.cms.label(ax=ax, label="Preliminary", data=False,
                          rlabel="", fontsize=14)
        except TypeError:
            hep.cms.label(ax=ax, text="Preliminary", data=False,
                          rlabel="", fontsize=14)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="Photon_E5_n*",
                        help="Glob pattern for eval_scaling subdirs (default: Photon_E5_n*)")
    parser.add_argument("--dataset", default="Photon_E5",
                        help="Dataset name in eval_results_all for model KS lookup (default: Photon_E5)")
    parser.add_argument("--xmax", type=float, default=None,
                        help="Maximum x-axis value (number of showers)")
    parser.add_argument("--logx", action="store_true",
                        help="Use logarithmic x-axis scale")
    parser.add_argument("--logy", action="store_true",
                        help="Use logarithmic y-axis scale")
    parser.add_argument("--ks", action="store_true",
                        help="Also produce KS metric plots (default: separation power only)")
    parser.add_argument("-f", "--train_file_list", default="",
                        help="Training file list (.txt) to count available showers at target energy")
    parser.add_argument("-d", "--train_dir", default="",
                        help="Directory to prepend to paths in train_file_list")
    parser.add_argument("-e", "--energy", type=float, default=None,
                        help="Target energy (GeV) for counting training showers")
    parser.add_argument("--windows", type=float, nargs="+", default=[0.006],
                        help="Fractional energy windows for counting training showers (default: 0.006)")
    parser.add_argument("-o", "--output", default="",
                        help="Output file stem — '_ks_<cat>.png/.pdf' and '_sep_<cat>.png/.pdf' "
                             "will be appended (default: show interactively)")
    flags = parser.parse_args()

    rows = load_scaling_data(flags.pattern)
    if not rows:
        print(f"No data found matching '{flags.pattern}' in {SCALING}")
        return

    ns = np.array([r["n_showers"] for r in rows])

    n_train = {}
    if flags.train_file_list and flags.energy is not None:
        windows = tuple(flags.windows)
        cache_path = SCALING / f"train_counts_{flags.dataset}_{flags.energy}GeV.txt"
        cached = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                w_str, n_str = line.split("=")
                cached[float(w_str)] = int(n_str)
        if all(w in cached for w in windows):
            n_train = {w: cached[w] for w in windows}
            print(f"\nLoaded training shower counts from {cache_path.name}")
        else:
            pct = ", ".join(f"{w*100:.4g}%" for w in windows)
            print(f"\nCounting training showers within {pct} of {flags.energy} GeV ...")
            n_train = count_training_showers(flags.train_file_list, flags.train_dir,
                                             flags.energy, windows=windows)
            # Merge new counts into any existing cache entries, then rewrite
            cached.update(n_train)
            cache_path.write_text("\n".join(f"{w}={n}" for w, n in cached.items()))
            print(f"  Saved counts to {cache_path.name}")
        for w, n in n_train.items():
            print(f"  {w*100:.4g}% window: {n:,} showers")

    metrics = [("Sep", "Sep. power", "sep")]
    if flags.ks:
        metrics.insert(0, ("KS", "KS", "ks"))
    for metric_key_prefix, metric_label, suffix in metrics:
        print(f"\n{metric_label} plots:")
        for cat in CATS:
            fig = make_panel(cat, metric_key_prefix, metric_label, flags, rows, ns, n_train)

            if flags.output:
                stem    = flags.output.removesuffix(".png").removesuffix(".pdf")
                out_png = stem + f"_{suffix}_{cat}.png"
                out_pdf = stem + f"_{suffix}_{cat}.pdf"
                fig.savefig(out_png, dpi=150, bbox_inches="tight")
                fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
                print(f"  Saved -> {out_png}, {out_pdf}")
                plt.close(fig)
            else:
                print(f"  {cat}")

    if not flags.output:
        plt.show()


if __name__ == "__main__":
    main()
