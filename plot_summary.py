#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import warnings
from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from plotting.plotting_utils import CMS_COLORS, add_experiment_label, apply_plot_style


SUMMARY_STYLE_PRESETS = {
    "single_plot": {
        "figsize": (8.0, 8.0),
        "height_ratios": (3.0, 1.0),
        "reference_color": "#000000",
        "reference_band_color": "#000000",
        "model_band_alpha": 0.2,
        "ratio_band_alpha": 0.2,
        "reference_band_alpha": 0.2,
        "line_width": 1.0,
        "reference_line_width": 1.0,
        "reference_alpha": 0.8,
        "model_alpha": 1.0,
        "legend_fontsize": 16.0,
        "label_fontsize": 16.0,
        "legend_loc": "upper right",
        "legend_ncol": 1,
        "legend_handlelength": 1.2,
        "ratio_ticks": (0.7, 1.0, 1.3),
        "ratio_guide_lines": (0.7, 1.3),
        "ratio_ylim": (0.5, 1.5),
        "ratio_pad_fraction": 0.10,
        "ratio_min_span": 0.10,
        "ratio_guard_low": 0.70,
        "ratio_guard_high": 1.30,
    },
    "paper": {
        "figsize": (12.0, 11.0),
        "height_ratios": (3.5, 1.0),
        "reference_color": "#202020",
        "reference_band_color": "#7f7f7f",
        "model_band_alpha": 0.0,
        "ratio_band_alpha": 0.0,
        "reference_band_alpha": 0.12,
        "line_width": 2.0,
        "reference_line_width": 3.5,
        "reference_alpha": 0.95,
        "model_alpha": 0.98,
        "legend_fontsize": 22,
        "label_fontsize": 24,
        "legend_loc": "upper right",
        "legend_ncol": None,
        "legend_handlelength": 2.0,
        "ratio_ticks": None,
        "ratio_guide_lines": None,
        "ratio_ylim": None,
        "ratio_pad_fraction": 0.10,
        "ratio_min_span": 0.10,
        "ratio_guard_low": 0.97,
        "ratio_guard_high": 1.03,
        "upper_ylim_headroom": 1.6,
    },
    "diagnostic": {
        "figsize": (12.0, 11.0),
        "height_ratios": (3.5, 1.0),
        "reference_color": "#202020",
        "reference_band_color": "#7f7f7f",
        "model_band_alpha": 0.08,
        "ratio_band_alpha": 0.10,
        "reference_band_alpha": 0.12,
        "line_width": 2.0,
        "reference_line_width": 3.0,
        "reference_alpha": 0.95,
        "model_alpha": 0.98,
        "legend_fontsize": 22,
        "label_fontsize": 24,
        "legend_loc": "upper right",
        "legend_ncol": None,
        "legend_handlelength": 2.0,
        "ratio_ticks": None,
        "ratio_guide_lines": None,
        "ratio_ylim": None,
        "ratio_pad_fraction": 0.12,
        "ratio_min_span": 0.12,
        "ratio_guard_low": 0.96,
        "ratio_guard_high": 1.04,
        "upper_ylim_headroom": 1.6,
    },
}


MODEL_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]


class SummaryModelSpec(object):
    def __init__(self, name, path, color=None, linestyle=None):
        self.name = name
        self.path = path
        self.color = color
        self.linestyle = linestyle


class SummaryPlotConfig(object):
    def __init__(
        self,
        models,
        output_dir="plots/summary/",
        feature_labels=None,
        cms_qualifier="Preliminary",
        data_label="",
        style=None,
        geom_file=None,
    ):
        self.models = models
        self.output_dir = output_dir
        self.feature_labels = feature_labels or {}
        self.cms_qualifier = cms_qualifier
        self.data_label = data_label
        self.style = style or SummaryStyleConfig()
        self.geom_file = geom_file


class SummaryStyleConfig(object):
    def __init__(
        self,
        preset="paper",
        figsize=(12.0, 11.0),
        height_ratios=(3.5, 1.0),
        reference_color="#202020",
        reference_band_color="#7f7f7f",
        model_band_alpha=0.0,
        ratio_band_alpha=0.0,
        reference_band_alpha=0.12,
        line_width=1.8,
        reference_line_width=3.0,
        reference_alpha=0.95,
        model_alpha=0.98,
        legend_fontsize=18,
        label_fontsize=22,
        legend_loc="upper right",
        legend_ncol=None,
        legend_handlelength=2.0,
        ratio_ticks=None,
        ratio_guide_lines=None,
        ratio_ylim=None,
        ratio_pad_fraction=0.10,
        ratio_min_span=0.10,
        ratio_guard_low=0.97,
        ratio_guard_high=1.03,
        upper_ylim_headroom=1.6,
    ):
        self.preset = preset
        self.figsize = figsize
        self.height_ratios = height_ratios
        self.reference_color = reference_color
        self.reference_band_color = reference_band_color
        self.model_band_alpha = model_band_alpha
        self.ratio_band_alpha = ratio_band_alpha
        self.reference_band_alpha = reference_band_alpha
        self.line_width = line_width
        self.reference_line_width = reference_line_width
        self.reference_alpha = reference_alpha
        self.model_alpha = model_alpha
        self.legend_fontsize = legend_fontsize
        self.label_fontsize = label_fontsize
        self.legend_loc = legend_loc
        self.legend_ncol = legend_ncol
        self.legend_handlelength = legend_handlelength
        self.ratio_ticks = ratio_ticks
        self.ratio_guide_lines = ratio_guide_lines
        self.ratio_ylim = ratio_ylim
        self.ratio_pad_fraction = ratio_pad_fraction
        self.ratio_min_span = ratio_min_span
        self.ratio_guard_low = ratio_guard_low
        self.ratio_guard_high = ratio_guard_high
        self.upper_ylim_headroom = upper_ylim_headroom


def dup(arr):
    return np.append(arr, arr[-1])


def parse_named_inputs(input_files_arg):
    entries = [entry.strip() for entry in input_files_arg.split(",") if entry.strip()]
    if len(entries) == 0:
        raise ValueError("--input_files is empty.")

    named_sources = []
    for entry in entries:
        if ":" not in entry:
            raise ValueError(
                f"Invalid --input_files entry '{entry}'. Expected format name:path."
            )
        name, path_spec = entry.split(":", 1)
        name = name.strip()
        path_spec = path_spec.strip()
        if len(name) == 0 or len(path_spec) == 0:
            raise ValueError(
                f"Invalid --input_files entry '{entry}'. Name and path must be non-empty."
            )
        named_sources.append(SummaryModelSpec(name=name, path=path_spec))

    return named_sources


def _resolve_relative_path(base_dir, path_spec):
    if os.path.isabs(path_spec):
        return path_spec
    return os.path.normpath(os.path.join(base_dir, path_spec))


def _normalize_feature_labels(raw_labels):
    if raw_labels is None:
        return {}
    if not isinstance(raw_labels, dict):
        raise ValueError("`feature_labels` must be an object mapping raw names to labels.")
    feature_labels = {}
    for key, value in raw_labels.items():
        feature_key = str(key).strip()
        feature_value = str(value).strip()
        if len(feature_key) == 0 or len(feature_value) == 0:
            raise ValueError("`feature_labels` keys and values must be non-empty strings.")
        feature_labels[feature_key] = feature_value
    return feature_labels


def _validate_color(name, color, allow_none=False):
    if color is None and allow_none:
        return None
    color = str(color).strip()
    if len(color) == 0:
        if allow_none:
            return None
        raise ValueError("`%s` must be a non-empty color string." % name)
    if not mcolors.is_color_like(color):
        raise ValueError("Invalid matplotlib color specification for `%s`: %s" % (name, color))
    return color


def _validate_model_color(color):
    return _validate_color("color", color, allow_none=True)


def _validate_positive_pair(name, values):
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("`%s` must be a list or tuple of length 2." % name)
    parsed = []
    for value in values:
        number = float(value)
        if number <= 0.0:
            raise ValueError("`%s` entries must be > 0." % name)
        parsed.append(number)
    return tuple(parsed)


def _validate_nonnegative(name, value):
    number = float(value)
    if number < 0.0:
        raise ValueError("`%s` must be >= 0." % name)
    return number


def _validate_unit_interval(name, value):
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise ValueError("`%s` must be between 0 and 1." % name)
    return number


def _validate_positive(name, value):
    number = float(value)
    if number <= 0.0:
        raise ValueError("`%s` must be > 0." % name)
    return number


def _validate_optional_positive_int(name, value):
    if value is None:
        return None
    number = int(value)
    if number <= 0:
        raise ValueError("`%s` must be a positive integer." % name)
    return number


def _validate_ratio_ylim(value):
    if value is None:
        return None
    ratio_ylim = _validate_positive_pair("ratio_ylim", value)
    if ratio_ylim[0] >= ratio_ylim[1]:
        raise ValueError("`ratio_ylim` must be in ascending order.")
    return ratio_ylim


def _validate_optional_tick_values(name, values):
    if values is None:
        return None
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError("`%s` must be a non-empty list or tuple if provided." % name)
    return tuple(float(value) for value in values)


def load_summary_style_config(raw_style):
    if raw_style is None:
        raw_style = {}
    if not isinstance(raw_style, dict):
        raise ValueError("`style` must be an object if provided.")

    preset = str(raw_style.get("preset", "paper")).strip().lower() or "paper"
    if preset not in SUMMARY_STYLE_PRESETS:
        raise ValueError(
            "Unknown style preset `%s`. Available presets: %s"
            % (preset, ", ".join(sorted(SUMMARY_STYLE_PRESETS.keys())))
        )

    style_payload = dict(SUMMARY_STYLE_PRESETS[preset])
    for key, value in raw_style.items():
        if key == "preset":
            continue
        style_payload[key] = value

    ratio_guard_low = _validate_positive("ratio_guard_low", style_payload["ratio_guard_low"])
    ratio_guard_high = _validate_positive("ratio_guard_high", style_payload["ratio_guard_high"])
    if ratio_guard_low >= ratio_guard_high:
        raise ValueError("`ratio_guard_low` must be smaller than `ratio_guard_high`.")

    return SummaryStyleConfig(
        preset=preset,
        figsize=_validate_positive_pair("figsize", style_payload["figsize"]),
        height_ratios=_validate_positive_pair("height_ratios", style_payload["height_ratios"]),
        reference_color=_validate_color("reference_color", style_payload["reference_color"]),
        reference_band_color=_validate_color(
            "reference_band_color", style_payload["reference_band_color"]
        ),
        model_band_alpha=_validate_unit_interval("model_band_alpha", style_payload["model_band_alpha"]),
        ratio_band_alpha=_validate_unit_interval("ratio_band_alpha", style_payload["ratio_band_alpha"]),
        reference_band_alpha=_validate_unit_interval(
            "reference_band_alpha", style_payload["reference_band_alpha"]
        ),
        line_width=_validate_positive("line_width", style_payload["line_width"]),
        reference_line_width=_validate_positive(
            "reference_line_width", style_payload["reference_line_width"]
        ),
        reference_alpha=_validate_unit_interval("reference_alpha", style_payload["reference_alpha"]),
        model_alpha=_validate_unit_interval("model_alpha", style_payload["model_alpha"]),
        legend_fontsize=_validate_positive("legend_fontsize", style_payload["legend_fontsize"]),
        label_fontsize=_validate_positive("label_fontsize", style_payload["label_fontsize"]),
        legend_loc=str(style_payload["legend_loc"]).strip() or "upper right",
        legend_ncol=_validate_optional_positive_int("legend_ncol", style_payload["legend_ncol"]),
        legend_handlelength=_validate_positive(
            "legend_handlelength", style_payload["legend_handlelength"]
        ),
        ratio_ticks=_validate_optional_tick_values("ratio_ticks", style_payload["ratio_ticks"]),
        ratio_guide_lines=_validate_optional_tick_values(
            "ratio_guide_lines", style_payload["ratio_guide_lines"]
        ),
        ratio_ylim=_validate_ratio_ylim(style_payload["ratio_ylim"]),
        ratio_pad_fraction=_validate_nonnegative(
            "ratio_pad_fraction", style_payload["ratio_pad_fraction"]
        ),
        ratio_min_span=_validate_positive("ratio_min_span", style_payload["ratio_min_span"]),
        ratio_guard_low=ratio_guard_low,
        ratio_guard_high=ratio_guard_high,
        upper_ylim_headroom=_validate_positive("upper_ylim_headroom", style_payload["upper_ylim_headroom"]),
    )


def load_summary_plot_config(config_path):
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as handle:
        payload = json.load(handle)

    models_payload = payload.get("models")
    if not isinstance(models_payload, list) or len(models_payload) == 0:
        raise ValueError("Summary config must define a non-empty `models` list.")

    config_dir = os.path.dirname(config_path)
    models = []
    for idx, model_payload in enumerate(models_payload):
        if not isinstance(model_payload, dict):
            raise ValueError(f"`models[{idx}]` must be an object.")

        name = str(model_payload.get("name", "")).strip()
        path = str(model_payload.get("path", "")).strip()
        if len(name) == 0 or len(path) == 0:
            raise ValueError(f"`models[{idx}]` must define non-empty `name` and `path`.")

        raw_ls = model_payload.get("linestyle")
        models.append(
            SummaryModelSpec(
                name=name,
                path=_resolve_relative_path(config_dir, path),
                color=_validate_model_color(model_payload.get("color")),
                linestyle=raw_ls,
            )
        )

    output_dir = str(payload.get("output_dir", "plots/summary/")).strip()
    if len(output_dir) == 0:
        raise ValueError("`output_dir` must be a non-empty string if provided.")
    output_dir = _resolve_relative_path(config_dir, output_dir)

    cms_qualifier = str(payload.get("cms_qualifier", "Preliminary")).strip() or "Preliminary"
    data_label = str(payload.get("data_label", "")).strip()
    feature_labels = _normalize_feature_labels(payload.get("feature_labels"))
    style = load_summary_style_config(payload.get("style"))
    geom_file_raw = payload.get("geom_file")
    geom_file = None
    if geom_file_raw:
        geom_file = _resolve_relative_path(config_dir, str(geom_file_raw).strip())
    return SummaryPlotConfig(
        models=models,
        output_dir=output_dir,
        feature_labels=feature_labels,
        cms_qualifier=cms_qualifier,
        data_label=data_label,
        style=style,
        geom_file=geom_file,
    )


def build_summary_plot_config(input_files_arg=None, config_path=None, output_dir=None):
    if config_path and input_files_arg:
        raise ValueError("Use either --config or --input_files, not both.")

    if config_path:
        config = load_summary_plot_config(config_path)
        if output_dir is not None:
            config = SummaryPlotConfig(
                models=config.models,
                output_dir=output_dir,
                feature_labels=config.feature_labels,
                cms_qualifier=config.cms_qualifier,
                data_label=config.data_label,
                style=config.style,
                geom_file=config.geom_file,
            )
        return config

    if not input_files_arg:
        raise ValueError("Either --config or --input_files is required.")

    return SummaryPlotConfig(
        models=parse_named_inputs(input_files_arg),
        output_dir=output_dir or "plots/summary/",
        feature_labels={},
        cms_qualifier="Preliminary",
        style=SummaryStyleConfig(),
    )


def resolve_summary_npz_files(path_spec):
    path_spec = path_spec.strip()

    if os.path.isdir(path_spec):
        return sorted(
            os.path.join(path_spec, fname)
            for fname in os.listdir(path_spec)
            if fname.endswith(".npz") and ".feat." not in fname
        )

    if (
        os.path.isfile(path_spec)
        and path_spec.endswith(".npz")
        and ".feat." not in os.path.basename(path_spec)
    ):
        return [path_spec]

    if os.path.isfile(path_spec):
        base_dir = os.path.dirname(path_spec)
        npz_files = []
        with open(path_spec, "r") as handle:
            for line in handle:
                fpath = line.strip()
                if len(fpath) == 0 or fpath.startswith("#"):
                    continue
                if not os.path.isabs(fpath):
                    rel_path = os.path.join(base_dir, fpath)
                    if os.path.exists(rel_path):
                        fpath = rel_path
                if (
                    fpath.endswith(".npz")
                    and ".feat." not in os.path.basename(fpath)
                    and os.path.isfile(fpath)
                ):
                    npz_files.append(fpath)
        return sorted(npz_files)

    glob_matches = sorted(glob.glob(path_spec))
    return [
        fpath
        for fpath in glob_matches
        if fpath.endswith(".npz") and ".feat." not in os.path.basename(fpath)
    ]


_OCCUPANCY_RE = re.compile(r"^OccupancyLayer(\d+)$", re.IGNORECASE)


def _load_layer_ncells(geom_file):
    """Load per-layer cell counts from a geometry pickle. Returns a 1D float array."""
    if not geom_file or not os.path.isfile(geom_file):
        return None
    import utils as _utils
    geom = _utils.load_geom(geom_file)
    return np.asarray(geom.ncells, dtype=np.float64)


def _occupancy_layer_index(feature_name):
    match = _OCCUPANCY_RE.match(feature_name)
    if match is None:
        return None
    return int(match.group(1))


def _convert_occupancy_payload(payload, ncells_layer):
    """Convert an occupancy histogram payload from active-cell counts to percent.

    Bin edges are scaled by 100/ncells (counts -> %), and densities are scaled
    by ncells/100 to preserve the integral (= 1).
    """
    if ncells_layer is None or ncells_layer <= 0:
        return payload
    k = 100.0 / float(ncells_layer)
    inv = 1.0 / k
    return {
        "binning": payload["binning"] * k,
        "dist_ref": payload["dist_ref"] * inv,
        "dist_ref_err": payload["dist_ref_err"] * inv,
        "dist_gen": payload["dist_gen"] * inv,
        "dist_gen_err": payload["dist_gen_err"] * inv,
    }


def load_histogram_npz(npz_file):
    with np.load(npz_file) as data:
        required = {"dist_ref", "dist_ref_err", "dist_gen", "dist_gen_err", "binning"}
        missing = required - set(data.keys())
        if len(missing) > 0:
            return None
        return {
            "dist_ref": np.asarray(data["dist_ref"], dtype=np.float64),
            "dist_ref_err": np.asarray(data["dist_ref_err"], dtype=np.float64),
            "dist_gen": np.asarray(data["dist_gen"], dtype=np.float64),
            "dist_gen_err": np.asarray(data["dist_gen_err"], dtype=np.float64),
            "binning": np.asarray(data["binning"], dtype=np.float64),
        }


def load_profile_npz(npz_file):
    with np.load(npz_file) as data:
        required = {"ref_mean", "ref_std", "ref_sem", "gen_mean", "gen_std", "gen_sem"}
        missing = required - set(data.keys())
        if len(missing) > 0:
            return None
        return {
            "ref_mean": np.asarray(data["ref_mean"], dtype=np.float64),
            "ref_sem": np.asarray(data["ref_sem"], dtype=np.float64),
            "gen_mean": np.asarray(data["gen_mean"], dtype=np.float64),
            "gen_sem": np.asarray(data["gen_sem"], dtype=np.float64),
        }


def default_feature_label(feature_name):
    label = feature_name.replace("_", " ")
    label = label.replace("Energyfraction", "Energy fraction ")
    label = label.replace("LongitudinalProfile", "Longitudinal profile")
    label = label.replace("TransverseProfile", "Transverse profile")
    label = label.replace("IncidentEnergy", "Incident energy")
    label = label.replace("ERatio", "E ratio")
    label = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", label)
    label = re.sub(r"([a-z])([A-Z])", r"\1 \2", label)
    label = re.sub(r"([A-Za-z])(\d)", r"\1 \2", label)
    label = re.sub(r"(\d)([A-Za-z])", r"\1 \2", label)
    label = re.sub(r"\s+", " ", label).strip()
    # Sentence-case post-fix: lowercase common words that came from
    # camelCase splits and should not be title-cased.
    label = label.replace("Energy Ratio", "Energy ratio")
    label = label.replace("Energy Fraction", "Energy fraction")
    label = label.replace("Incident Energy", "Incident energy")
    # Add units: X/Y center & width are in cm; occupancy is in % (post-converted)
    name_lower = feature_name.lower()
    if re.match(r"^[xy](center|width)layer\d+$", name_lower):
        label = re.sub(
            r"\b([XY]) (Center|Width)\b",
            lambda m: f"{m.group(1)} {m.group(2).lower()} [cm]",
            label,
        )
    elif name_lower.startswith("occupancylayer"):
        label = re.sub(r"\bOccupancy\b", "Occupancy [%]", label)
    elif name_lower == "incidentenergy":
        label = label + " [GeV]"
    # Prepend a comma before trailing "layer N" / "ring N" (kept lowercase).
    label = re.sub(
        r"\s*\b([Ll]ayer|[Rr]ing) (\d+)$",
        lambda m: f", {m.group(1).lower()} {m.group(2)}",
        label,
    )
    return label


def get_feature_label(feature_name, feature_labels):
    if feature_labels and feature_name in feature_labels:
        return feature_labels[feature_name]
    return default_feature_label(feature_name)


def resolve_model_colors(models):
    fallback_colors = list(CMS_COLORS)
    if len(models) > len(fallback_colors):
        fallback_colors = [
            mcolors.to_hex(color) for color in plt.cm.tab20(np.linspace(0, 1, len(models)))
        ]

    resolved = {}
    for idx, model in enumerate(models):
        resolved[model.name] = model.color or fallback_colors[idx]
    return resolved


def resolve_model_linestyles(models):
    resolved = {}
    for idx, model in enumerate(models):
        resolved[model.name] = model.linestyle or MODEL_LINESTYLES[idx % len(MODEL_LINESTYLES)]
    return resolved


def _ratio_window(dist_ref, dist_ref_err, ratio, ratio_err):
    finite_mask = np.isfinite(ratio) & np.isfinite(ratio_err)
    stable_mask = finite_mask & (dist_ref > 3.0 * dist_ref_err)
    mask = stable_mask if np.any(stable_mask) else finite_mask
    if not np.any(mask):
        return None
    return (
        float(np.nanmin(ratio[mask] - ratio_err[mask])),
        float(np.nanmax(ratio[mask] + ratio_err[mask])),
    )


def _legend_columns(n_models):
    return 1 if n_models <= 4 else 2


def _setup_summary_axes(style, summary_config):
    """Create figure with upper + ratio panel and apply common formatting."""
    fig, ax = plt.subplots(
        2, 1,
        figsize=style.figsize,
        gridspec_kw={"hspace": 0.12, "height_ratios": style.height_ratios},
        sharex=True,
    )
    return fig, ax


def _finish_summary_axes(fig, ax, style, summary_config, feature_name, n_models,
                         ratio_min, ratio_max, models_for_ratio_label=None):
    """Apply common labels, legend, experiment label, and ratio y-limits.

    ``models_for_ratio_label`` is the list of (model_name, payload) tuples used
    to render the ratio panel. When given and a single model is plotted, the
    ratio label uses ``Model / Geant4`` with that model's name.
    """
    rlabel = summary_config.data_label if summary_config.data_label else "Phase-II"
    ax[0].set_ylim(0.0, None)
    # Add headroom so legend/CMS label don't overlap data
    _, ymax = ax[0].get_ylim()
    ax[0].set_ylim(0.0, ymax * style.upper_ylim_headroom)
    ax[0].set_ylabel("Arbitrary units", fontsize=style.label_fontsize, loc="top")
    ax[1].set_xlabel(
        get_feature_label(feature_name, summary_config.feature_labels),
        fontsize=style.label_fontsize,
    )
    if models_for_ratio_label and len(models_for_ratio_label) == 1:
        single_name = models_for_ratio_label[0][0]
        ratio_label = r"$\frac{\text{%s}}{\text{Geant4}}$" % single_name
    else:
        ratio_label = "Model / Geant4"
    ax[1].set_ylabel(ratio_label, fontsize=max(style.label_fontsize - 2.0, 1.0),
                     loc="center")

    # Align tick label sizes with axis labels
    tick_fontsize = max(style.label_fontsize - 4.0, 12.0)
    for a in ax:
        a.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        a.tick_params(axis="both", which="minor", labelsize=tick_fontsize - 2.0)

    # Light grid on ratio panel for readability
    ax[1].yaxis.grid(True, which="major", color="#cccccc", linewidth=0.5, alpha=0.6)
    ax[1].set_axisbelow(True)

    ax[0].legend(
        loc=style.legend_loc,
        ncol=style.legend_ncol or _legend_columns(n_models + 1),
        frameon=False,
        fontsize=style.legend_fontsize,
        handlelength=getattr(style, "legend_handlelength", 3.0),
        columnspacing=1.0,
    )
    add_experiment_label(ax[0], label=summary_config.cms_qualifier, rlabel=rlabel)

    if style.ratio_ylim is not None:
        ax[1].set_ylim(style.ratio_ylim[0], style.ratio_ylim[1])
    else:
        if not np.isfinite(ratio_min) or not np.isfinite(ratio_max):
            ratio_min, ratio_max = 0.92, 1.08
        # Symmetric around 1.0: find max deviation, add padding
        dev_lo = max(1.0 - ratio_min, 1.0 - style.ratio_guard_low)
        dev_hi = max(ratio_max - 1.0, style.ratio_guard_high - 1.0)
        dev = max(dev_lo, dev_hi)
        dev *= (1.0 + style.ratio_pad_fraction)
        dev = max(dev, style.ratio_min_span / 2.0)
        # Hard cap: never exceed [0, 2]
        dev = min(dev, 1.0)
        ax[1].set_ylim(1.0 - dev, 1.0 + dev)

    if style.ratio_ticks is not None:
        ax[1].set_yticks(style.ratio_ticks)
    if style.ratio_guide_lines is not None:
        for guide_y in style.ratio_guide_lines:
            ax[1].axhline(guide_y, c=style.reference_color, ls="--", lw=0.5,
                          alpha=0.8)

    fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.10)


def _save_summary_fig(fig, output_dir, feature_name, upper_ylim_headroom=1.4):
    output_base = os.path.join(output_dir, f"summary_{feature_name}")
    ax0 = fig.axes[0]
    # Save linear version
    fig.savefig(output_base + ".png", dpi=150)
    fig.savefig(output_base + ".pdf", dpi=300, format="pdf")
    # Save log-y version
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ax0.set_yscale("log")
    ax0.set_ylim(auto=True)
    ax0.relim()
    ax0.autoscale_view()
    ymin, ymax = ax0.get_ylim()
    # In log scale, headroom needs to be applied in log-space
    # e.g. headroom=1.6 means extend by 60% of the log-range
    if ymax > 0 and ymin > 0:
        log_range = np.log10(ymax) - np.log10(ymin)
        ax0.set_ylim(ymin, 10 ** (np.log10(ymax) + log_range * (upper_ylim_headroom - 1.0)))
    else:
        ax0.set_ylim(ymin, ymax * upper_ylim_headroom)
    fig.savefig(output_base + "_logy.png", dpi=150)
    fig.savefig(output_base + "_logy.pdf", dpi=300, format="pdf")
    plt.close(fig)


def make_summary_plots(summary_config):
    apply_plot_style()
    os.makedirs(summary_config.output_dir, exist_ok=True)

    style = summary_config.style
    model_sources = summary_config.models
    model_colors = resolve_model_colors(model_sources)
    model_linestyles = resolve_model_linestyles(model_sources)
    model_feature_files = {}

    for model_spec in model_sources:
        if model_spec.name in model_feature_files:
            raise ValueError(f"Duplicate model name in summary config: {model_spec.name}")
        npz_files = resolve_summary_npz_files(model_spec.path)
        if len(npz_files) == 0:
            raise ValueError(
                f"No histogram npz files found for {model_spec.name}:{model_spec.path}"
            )

        feature_map = {}
        for npz_file in npz_files:
            feature_key = os.path.splitext(os.path.basename(npz_file))[0]
            feature_map[feature_key] = npz_file
        model_feature_files[model_spec.name] = feature_map

    common_features = sorted(
        set.intersection(*(set(feature_map.keys()) for feature_map in model_feature_files.values()))
    )
    if len(common_features) == 0:
        raise ValueError("No common feature names found across summary plot inputs.")

    layer_ncells = _load_layer_ncells(summary_config.geom_file)
    if layer_ncells is None and summary_config.geom_file:
        print(f"WARNING: geom_file '{summary_config.geom_file}' not found; "
              "occupancy will remain in raw cell counts.")

    # Separate histogram features from profile features
    profile_features = []
    histogram_features = []
    for feature_name in common_features:
        # Try loading as profile first from the first model
        first_model = model_sources[0].name
        npz_file = model_feature_files[first_model][feature_name]
        profile_payload = load_profile_npz(npz_file)
        if profile_payload is not None:
            profile_features.append(feature_name)
        else:
            histogram_features.append(feature_name)

    # --- Histogram overlay plots ---
    for feature_name in histogram_features:
        loaded = []
        binning_ref = None
        geant_ref = None
        geant_ref_err = None
        skip_feature = False

        # Detect occupancy features and pre-compute the per-layer cell count
        occ_layer = _occupancy_layer_index(feature_name)
        occ_ncells = None
        if occ_layer is not None and layer_ncells is not None:
            if 0 <= occ_layer < layer_ncells.size:
                occ_ncells = float(layer_ncells[occ_layer])

        for model_spec in model_sources:
            model_name = model_spec.name
            npz_file = model_feature_files[model_name][feature_name]
            payload = load_histogram_npz(npz_file)
            if payload is None:
                print(
                    f"Skipping {feature_name}: not a histogram npz ({npz_file})"
                )
                skip_feature = True
                break
            if occ_ncells is not None:
                payload = _convert_occupancy_payload(payload, occ_ncells)
            if binning_ref is None:
                binning_ref = payload["binning"]
                geant_ref = payload["dist_ref"]
                geant_ref_err = payload["dist_ref_err"]
            elif not np.allclose(payload["binning"], binning_ref, rtol=1e-8, atol=1e-12):
                print(
                    f"Skipping {feature_name}: incompatible binning for model {model_name} "
                    f"({npz_file})"
                )
                skip_feature = True
                break
            elif not np.allclose(payload["dist_ref"], geant_ref, rtol=1e-8, atol=1e-12):
                print(
                    f"Skipping {feature_name}: incompatible Geant4 reference for model "
                    f"{model_name} ({npz_file})"
                )
                skip_feature = True
                break
            elif not np.allclose(payload["dist_ref_err"], geant_ref_err, rtol=1e-8, atol=1e-12, equal_nan=True):
                print(
                    f"Skipping {feature_name}: incompatible Geant4 reference errors for "
                    f"model {model_name} ({npz_file})"
                )
                skip_feature = True
                break
            loaded.append((model_name, payload))

        if skip_feature:
            continue

        fig, ax = _setup_summary_axes(style, summary_config)

        with np.errstate(divide="ignore", invalid="ignore"):
            geant_ratio_err = np.divide(
                geant_ref_err,
                geant_ref,
                out=np.zeros_like(geant_ref_err),
                where=geant_ref > 0.0,
            )

        ax[0].step(
            binning_ref,
            dup(geant_ref),
            label="Geant4",
            linestyle="-",
            alpha=style.reference_alpha,
            linewidth=style.reference_line_width,
            color=style.reference_color,
            where="post",
        )
        if style.reference_band_alpha > 0.0:
            ax[0].fill_between(
                binning_ref,
                dup(geant_ref + geant_ref_err),
                dup(np.maximum(geant_ref - geant_ref_err, 0.0)),
                step="post",
                color=style.reference_band_color,
                alpha=style.reference_band_alpha,
            )
            ax[1].fill_between(
                binning_ref,
                dup(np.maximum(1.0 - geant_ratio_err, 0.0)),
                dup(1.0 + geant_ratio_err),
                step="post",
                color=style.reference_band_color,
                alpha=style.reference_band_alpha,
            )

        ratio_min = np.inf
        ratio_max = -np.inf
        for model_name, payload in loaded:
            color = model_colors[model_name]
            ls = model_linestyles[model_name]

            ax[0].step(
                binning_ref,
                dup(payload["dist_gen"]),
                label=model_name,
                where="post",
                linewidth=style.line_width,
                alpha=style.model_alpha,
                color=color,
                linestyle=ls,
            )
            if style.model_band_alpha > 0.0:
                ax[0].fill_between(
                    binning_ref,
                    dup(payload["dist_gen"] + payload["dist_gen_err"]),
                    dup(np.maximum(payload["dist_gen"] - payload["dist_gen_err"], 0.0)),
                    step="post",
                    color=color,
                    alpha=style.model_band_alpha,
                )

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.divide(
                    payload["dist_gen"],
                    payload["dist_ref"],
                    out=np.ones_like(payload["dist_gen"]),
                    where=payload["dist_ref"] > 0.0,
                )
                ratio_err = np.divide(
                    payload["dist_gen_err"],
                    payload["dist_ref"],
                    out=np.zeros_like(payload["dist_gen_err"]),
                    where=payload["dist_ref"] > 0.0,
                )

            ax[1].step(
                binning_ref,
                dup(ratio),
                where="post",
                linewidth=style.line_width,
                alpha=style.model_alpha,
                color=color,
                linestyle=ls,
            )
            if style.ratio_band_alpha > 0.0:
                ax[1].fill_between(
                    binning_ref,
                    dup(ratio - ratio_err),
                    dup(ratio + ratio_err),
                    step="post",
                    color=color,
                    alpha=style.ratio_band_alpha,
                )

            ratio_window = _ratio_window(
                payload["dist_ref"], payload["dist_ref_err"], ratio, ratio_err
            )
            if ratio_window is not None:
                ratio_min = min(ratio_min, ratio_window[0])
                ratio_max = max(ratio_max, ratio_window[1])

        ax[1].hlines(
            1.0,
            binning_ref[0],
            binning_ref[-1],
            linewidth=1.0,
            alpha=style.reference_alpha,
            linestyle="-",
            color=style.reference_color,
        )
        if style.ratio_guide_lines is not None:
            for guide_y in style.ratio_guide_lines:
                ax[1].axhline(guide_y, c=style.reference_color, ls="--", lw=0.5, alpha=0.8)
        ax[0].set_xlim(binning_ref[0], binning_ref[-1])

        _finish_summary_axes(fig, ax, style, summary_config, feature_name,
                             len(loaded), ratio_min, ratio_max,
                             models_for_ratio_label=loaded)
        _save_summary_fig(fig, summary_config.output_dir, feature_name, style.upper_ylim_headroom)

    # --- Profile overlay plots ---
    for feature_name in profile_features:
        loaded_profiles = []
        ref_mean_ref = None
        skip_feature = False

        for model_spec in model_sources:
            model_name = model_spec.name
            npz_file = model_feature_files[model_name][feature_name]
            payload = load_profile_npz(npz_file)
            if payload is None:
                print(f"Skipping profile {feature_name}: missing keys ({npz_file})")
                skip_feature = True
                break
            if ref_mean_ref is None:
                ref_mean_ref = payload["ref_mean"]
            elif not np.allclose(payload["ref_mean"], ref_mean_ref, rtol=1e-6, atol=1e-12):
                print(
                    f"Skipping profile {feature_name}: incompatible Geant4 reference for "
                    f"model {model_name}"
                )
                skip_feature = True
                break
            loaded_profiles.append((model_name, payload))

        if skip_feature or len(loaded_profiles) == 0:
            continue

        ref_payload = loaded_profiles[0][1]
        ref_mean = ref_payload["ref_mean"]
        ref_sem = ref_payload["ref_sem"]
        n_bins = len(ref_mean)
        x = np.arange(n_bins)

        fig, ax = _setup_summary_axes(style, summary_config)

        # Geant4 reference
        ax[0].step(x, ref_mean, where="mid", color=style.reference_color,
                   linewidth=style.reference_line_width,
                   alpha=style.reference_alpha, label="Geant4",
                   linestyle="-")
        ax[0].fill_between(x, ref_mean - ref_sem, ref_mean + ref_sem,
                           alpha=style.reference_band_alpha, color=style.reference_band_color,
                           step="mid")

        with np.errstate(divide="ignore", invalid="ignore"):
            ref_ratio_err = np.where(ref_mean > 0, ref_sem / ref_mean, 0.0)
        ax[1].fill_between(x, 1 - ref_ratio_err, 1 + ref_ratio_err,
                           alpha=style.reference_band_alpha, color=style.reference_band_color,
                           step="mid")

        # Mask bins where reference is near-zero to avoid ratio explosions
        ref_threshold = 1e-3 * np.max(ref_mean) if np.max(ref_mean) > 0 else 0.0
        stable_bins = ref_mean > ref_threshold

        ratio_min = np.inf
        ratio_max = -np.inf
        for model_name, payload in loaded_profiles:
            color = model_colors[model_name]
            ls = model_linestyles[model_name]
            gen_mean = payload["gen_mean"]
            gen_sem = payload["gen_sem"]

            ax[0].step(x, gen_mean, where="mid", color=color,
                       linewidth=style.line_width, alpha=style.model_alpha,
                       label=model_name, linestyle=ls)
            if style.model_band_alpha > 0.0:
                ax[0].fill_between(x, gen_mean - gen_sem, gen_mean + gen_sem,
                                   alpha=style.model_band_alpha, color=color, step="mid")

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(ref_mean > 0, gen_mean / ref_mean, 1.0)
                ratio_err = np.where(ref_mean > 0, gen_sem / ref_mean, 0.0)

            ax[1].step(x, ratio, where="mid", color=color,
                       linewidth=style.line_width, alpha=style.model_alpha,
                       linestyle=ls)
            if style.ratio_band_alpha > 0.0:
                ax[1].fill_between(x, ratio - ratio_err, ratio + ratio_err,
                                   alpha=style.ratio_band_alpha, color=color, step="mid")

            finite = np.isfinite(ratio) & np.isfinite(ratio_err) & stable_bins
            if np.any(finite):
                ratio_min = min(ratio_min, float(np.nanmin(ratio[finite] - ratio_err[finite])))
                ratio_max = max(ratio_max, float(np.nanmax(ratio[finite] + ratio_err[finite])))

        ax[1].axhline(1.0, color=style.reference_color, linewidth=1.0, alpha=0.8)
        ax[0].set_xlim(-0.5, n_bins - 0.5)

        _finish_summary_axes(fig, ax, style, summary_config, feature_name,
                             len(loaded_profiles), ratio_min, ratio_max)
        # Override axis labels for profiles
        if "Longitudinal" in feature_name:
            ax[1].set_xlabel("Calorimeter Layer", fontsize=style.label_fontsize)
        elif "Transverse" in feature_name:
            ax[1].set_xlabel("Ring Number", fontsize=style.label_fontsize)
        ax[0].set_ylabel("Avg. energy fraction", fontsize=style.label_fontsize, loc="top")
        _save_summary_fig(fig, summary_config.output_dir, feature_name, style.upper_ylim_headroom)

    print(f"Summary plots saved in {summary_config.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay model histogram ratios (model/Geant4) from hgcal_metrics .npz outputs."
    )
    parser.add_argument(
        "--input_files",
        default=None,
        help=(
            "Inputs formatted as name1:path1,name2:path2,... "
            "where each path can be a directory of hist npz files, "
            "a single hist npz file, or a text file with one npz path per line."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional JSON config for summary plots. "
            "Defines models with name/path/color and can also set output_dir."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="Folder where summary plots are written. Overrides the config file if provided.",
    )
    args = parser.parse_args()
    summary_config = build_summary_plot_config(
        input_files_arg=args.input_files,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    make_summary_plots(summary_config)


if __name__ == "__main__":
    main()
