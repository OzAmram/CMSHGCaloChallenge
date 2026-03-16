#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from plotting.plotting_utils import CMS_COLORS, add_experiment_label, apply_plot_style


SUMMARY_STYLE_PRESETS = {
    "paper": {
        "figsize": (7.2, 5.0),
        "height_ratios": (3.5, 1.0),
        "reference_color": "#202020",
        "reference_band_color": "#7f7f7f",
        "model_band_alpha": 0.0,
        "ratio_band_alpha": 0.0,
        "reference_band_alpha": 0.12,
        "line_width": 1.8,
        "reference_line_width": 2.0,
        "legend_fontsize": 10.5,
        "label_fontsize": 14.0,
        "legend_loc": "upper right",
        "legend_ncol": None,
        "ratio_ylim": None,
        "ratio_pad_fraction": 0.10,
        "ratio_min_span": 0.10,
        "ratio_guard_low": 0.97,
        "ratio_guard_high": 1.03,
    },
    "diagnostic": {
        "figsize": (7.2, 5.4),
        "height_ratios": (3.2, 1.15),
        "reference_color": "#202020",
        "reference_band_color": "#7f7f7f",
        "model_band_alpha": 0.08,
        "ratio_band_alpha": 0.10,
        "reference_band_alpha": 0.12,
        "line_width": 1.5,
        "reference_line_width": 1.7,
        "legend_fontsize": 11.0,
        "label_fontsize": 14.0,
        "legend_loc": "upper right",
        "legend_ncol": None,
        "ratio_ylim": None,
        "ratio_pad_fraction": 0.12,
        "ratio_min_span": 0.12,
        "ratio_guard_low": 0.96,
        "ratio_guard_high": 1.04,
    },
}


class SummaryModelSpec(object):
    def __init__(self, name, path, color=None):
        self.name = name
        self.path = path
        self.color = color


class SummaryPlotConfig(object):
    def __init__(
        self,
        models,
        output_dir="plots/summary/",
        feature_labels=None,
        cms_qualifier="Simulation",
        style=None,
    ):
        self.models = models
        self.output_dir = output_dir
        self.feature_labels = feature_labels or {}
        self.cms_qualifier = cms_qualifier
        self.style = style or SummaryStyleConfig()


class SummaryStyleConfig(object):
    def __init__(
        self,
        preset="paper",
        figsize=(7.2, 5.0),
        height_ratios=(3.5, 1.0),
        reference_color="#202020",
        reference_band_color="#7f7f7f",
        model_band_alpha=0.0,
        ratio_band_alpha=0.0,
        reference_band_alpha=0.12,
        line_width=1.8,
        reference_line_width=2.0,
        legend_fontsize=10.5,
        label_fontsize=14.0,
        legend_loc="upper right",
        legend_ncol=None,
        ratio_ylim=None,
        ratio_pad_fraction=0.10,
        ratio_min_span=0.10,
        ratio_guard_low=0.97,
        ratio_guard_high=1.03,
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
        self.legend_fontsize = legend_fontsize
        self.label_fontsize = label_fontsize
        self.legend_loc = legend_loc
        self.legend_ncol = legend_ncol
        self.ratio_ylim = ratio_ylim
        self.ratio_pad_fraction = ratio_pad_fraction
        self.ratio_min_span = ratio_min_span
        self.ratio_guard_low = ratio_guard_low
        self.ratio_guard_high = ratio_guard_high


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
        legend_fontsize=_validate_positive("legend_fontsize", style_payload["legend_fontsize"]),
        label_fontsize=_validate_positive("label_fontsize", style_payload["label_fontsize"]),
        legend_loc=str(style_payload["legend_loc"]).strip() or "upper right",
        legend_ncol=_validate_optional_positive_int("legend_ncol", style_payload["legend_ncol"]),
        ratio_ylim=_validate_ratio_ylim(style_payload["ratio_ylim"]),
        ratio_pad_fraction=_validate_nonnegative(
            "ratio_pad_fraction", style_payload["ratio_pad_fraction"]
        ),
        ratio_min_span=_validate_positive("ratio_min_span", style_payload["ratio_min_span"]),
        ratio_guard_low=ratio_guard_low,
        ratio_guard_high=ratio_guard_high,
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

        models.append(
            SummaryModelSpec(
                name=name,
                path=_resolve_relative_path(config_dir, path),
                color=_validate_model_color(model_payload.get("color")),
            )
        )

    output_dir = str(payload.get("output_dir", "plots/summary/")).strip()
    if len(output_dir) == 0:
        raise ValueError("`output_dir` must be a non-empty string if provided.")
    output_dir = _resolve_relative_path(config_dir, output_dir)

    cms_qualifier = str(payload.get("cms_qualifier", "Simulation")).strip() or "Simulation"
    feature_labels = _normalize_feature_labels(payload.get("feature_labels"))
    style = load_summary_style_config(payload.get("style"))
    return SummaryPlotConfig(
        models=models,
        output_dir=output_dir,
        feature_labels=feature_labels,
        cms_qualifier=cms_qualifier,
        style=style,
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
                style=config.style,
            )
        return config

    if not input_files_arg:
        raise ValueError("Either --config or --input_files is required.")

    return SummaryPlotConfig(
        models=parse_named_inputs(input_files_arg),
        output_dir=output_dir or "plots/summary/",
        feature_labels={},
        cms_qualifier="Simulation",
        style=SummaryStyleConfig(),
    )


def resolve_summary_npz_files(path_spec):
    path_spec = path_spec.strip()

    if os.path.isdir(path_spec):
        return sorted(
            os.path.join(path_spec, fname)
            for fname in os.listdir(path_spec)
            if fname.endswith(".npz") and not fname.endswith(".feat.npz")
        )

    if (
        os.path.isfile(path_spec)
        and path_spec.endswith(".npz")
        and not path_spec.endswith(".feat.npz")
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
                    and not fpath.endswith(".feat.npz")
                    and os.path.isfile(fpath)
                ):
                    npz_files.append(fpath)
        return sorted(npz_files)

    glob_matches = sorted(glob.glob(path_spec))
    return [
        fpath
        for fpath in glob_matches
        if fpath.endswith(".npz") and not fpath.endswith(".feat.npz")
    ]


def load_histogram_npz(npz_file):
    with np.load(npz_file) as data:
        required = {"dist_ref", "dist_ref_err", "dist_gen", "dist_gen_err", "binning"}
        missing = required - set(data.keys())
        if len(missing) > 0:
            raise KeyError(
                f"Missing keys {sorted(missing)} in {npz_file}. "
                "Expected output from plotting.plotting_utils.make_hist."
            )
        return {
            "dist_ref": np.asarray(data["dist_ref"], dtype=np.float64),
            "dist_ref_err": np.asarray(data["dist_ref_err"], dtype=np.float64),
            "dist_gen": np.asarray(data["dist_gen"], dtype=np.float64),
            "dist_gen_err": np.asarray(data["dist_gen_err"], dtype=np.float64),
            "binning": np.asarray(data["binning"], dtype=np.float64),
        }


def default_feature_label(feature_name):
    label = feature_name.replace("_", " ")
    label = label.replace("Energyfraction", "Energy fraction")
    label = label.replace("LongitudinalProfile", "Longitudinal Profile")
    label = label.replace("TransverseProfile", "Transverse Profile")
    label = label.replace("IncidentE", "Incident E")
    label = label.replace("ERatio", "E Ratio")
    label = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", label)
    label = re.sub(r"([a-z])([A-Z])", r"\1 \2", label)
    label = re.sub(r"([A-Za-z])(\d)", r"\1 \2", label)
    label = re.sub(r"(\d)([A-Za-z])", r"\1 \2", label)
    return re.sub(r"\s+", " ", label).strip()


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


def make_summary_plots(summary_config):
    apply_plot_style()
    os.makedirs(summary_config.output_dir, exist_ok=True)

    style = summary_config.style
    model_sources = summary_config.models
    model_colors = resolve_model_colors(model_sources)
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

    for feature_name in common_features:
        loaded = []
        binning_ref = None
        geant_ref = None
        geant_ref_err = None
        skip_feature = False

        for model_spec in model_sources:
            model_name = model_spec.name
            npz_file = model_feature_files[model_name][feature_name]
            payload = load_histogram_npz(npz_file)
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
            elif not np.allclose(payload["dist_ref_err"], geant_ref_err, rtol=1e-8, atol=1e-12):
                print(
                    f"Skipping {feature_name}: incompatible Geant4 reference errors for "
                    f"model {model_name} ({npz_file})"
                )
                skip_feature = True
                break
            loaded.append((model_name, payload))

        if skip_feature:
            continue

        fig, ax = plt.subplots(
            2,
            1,
            figsize=style.figsize,
            gridspec_kw={"hspace": 0.0, "height_ratios": style.height_ratios},
            sharex=True,
        )

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
            alpha=0.95,
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

            ax[0].step(
                binning_ref,
                dup(payload["dist_gen"]),
                label=model_name,
                where="post",
                linewidth=style.line_width,
                alpha=0.98,
                color=color,
                linestyle="-",
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
                alpha=0.98,
                color=color,
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
            alpha=0.8,
            linestyle="-",
            color=style.reference_color,
        )
        ax[0].set_xlim(binning_ref[0], binning_ref[-1])
        ax[0].set_ylim(0.0, None)
        ax[0].set_ylabel("a.u.", fontsize=style.label_fontsize)
        ax[1].set_xlabel(
            get_feature_label(feature_name, summary_config.feature_labels),
            fontsize=style.label_fontsize,
        )
        ax[1].set_ylabel("Ratio to Geant4", fontsize=max(style.label_fontsize - 1.0, 1.0))
        ax[0].legend(
            loc=style.legend_loc,
            ncol=style.legend_ncol or _legend_columns(len(loaded) + 1),
            frameon=False,
            fontsize=style.legend_fontsize,
            handlelength=2.0,
            columnspacing=1.0,
        )
        add_experiment_label(ax[0], label=summary_config.cms_qualifier)

        if style.ratio_ylim is not None:
            ax[1].set_ylim(style.ratio_ylim[0], style.ratio_ylim[1])
        else:
            if not np.isfinite(ratio_min) or not np.isfinite(ratio_max):
                ratio_min, ratio_max = 0.92, 1.08
            ratio_min = min(ratio_min, style.ratio_guard_low)
            ratio_max = max(ratio_max, style.ratio_guard_high)
            ratio_pad = style.ratio_pad_fraction * max(
                ratio_max - ratio_min, style.ratio_min_span
            )
            ax[1].set_ylim(max(0.0, ratio_min - ratio_pad), ratio_max + ratio_pad)

        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
        output_base = os.path.join(summary_config.output_dir, f"summary_{feature_name}")
        fig.savefig(output_base + ".png")
        fig.savefig(output_base + ".pdf", dpi=300, format="pdf")
        plt.close(fig)

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
