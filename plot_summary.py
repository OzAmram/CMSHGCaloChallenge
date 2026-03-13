#!/usr/bin/env python3
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

import utils
from plotting.plotting_utils import set_cms_style, cms_label, CMS_COLORS


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
        named_sources.append((name, path_spec))

    return named_sources


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


def make_summary_plots(input_files_arg, output_dir):
    set_cms_style()
    os.makedirs(output_dir, exist_ok=True)

    model_sources = parse_named_inputs(input_files_arg)
    model_feature_files = {}

    for model_name, source in model_sources:
        npz_files = resolve_summary_npz_files(source)
        if len(npz_files) == 0:
            raise ValueError(f"No histogram npz files found for {model_name}:{source}")

        feature_map = {}
        for npz_file in npz_files:
            feature_key = os.path.splitext(os.path.basename(npz_file))[0]
            feature_map[feature_key] = npz_file
        model_feature_files[model_name] = feature_map

    common_features = sorted(
        set.intersection(*(set(feature_map.keys()) for feature_map in model_feature_files.values()))
    )
    if len(common_features) == 0:
        raise ValueError("No common feature names found across --input_files inputs.")

    n_models = len(model_feature_files)
    if n_models <= len(CMS_COLORS):
        colors = CMS_COLORS[:n_models]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_models))

    for feature_name in common_features:
        loaded = []
        binning_ref = None
        geant_ref = None
        geant_ref_err = None
        skip_feature = False

        for model_name in model_feature_files:
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
            loaded.append((model_name, payload))

        if skip_feature:
            continue

        fig, ax = plt.subplots(
            2,
            1,
            figsize=(6.2, 5.2),
            gridspec_kw={"hspace": 0.0, "height_ratios": (3, 1)},
            sharex=True,
        )

        ax[0].step(
            binning_ref,
            dup(geant_ref),
            label="Geant4",
            linestyle="-",
            alpha=0.9,
            linewidth=1.2,
            color="k",
            where="post",
        )
        ax[0].fill_between(
            binning_ref,
            dup(geant_ref + geant_ref_err),
            dup(np.maximum(geant_ref - geant_ref_err, 0.0)),
            step="post",
            color="k",
            alpha=0.2,
        )

        ratio_min = np.inf
        ratio_max = -np.inf
        for idx, (model_name, payload) in enumerate(loaded):
            color = colors[idx % len(colors)]

            ax[0].step(
                binning_ref,
                dup(payload["dist_gen"]),
                label=model_name,
                where="post",
                linewidth=1.2,
                alpha=1.0,
                color=color,
                linestyle="-",
            )
            ax[0].fill_between(
                binning_ref,
                dup(payload["dist_gen"] + payload["dist_gen_err"]),
                dup(np.maximum(payload["dist_gen"] - payload["dist_gen_err"], 0.0)),
                step="post",
                color=color,
                alpha=0.2,
            )

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
                linewidth=1.2,
                alpha=1.0,
                color=color,
            )
            ax[1].fill_between(
                binning_ref,
                dup(ratio - ratio_err),
                dup(ratio + ratio_err),
                step="post",
                color=color,
                alpha=0.2,
            )

            ratio_min = min(ratio_min, np.nanmin(ratio - ratio_err))
            ratio_max = max(ratio_max, np.nanmax(ratio + ratio_err))

        ax[1].hlines(
            1.0,
            binning_ref[0],
            binning_ref[-1],
            linewidth=1.0,
            alpha=0.8,
            linestyle="-",
            color="k",
        )
        ax[0].set_xlim(binning_ref[0], binning_ref[-1])
        ax[0].set_ylim(0.0, None)
        ax[0].set_ylabel("a.u.")
        ax[1].set_xlabel(feature_name)
        ax[1].set_ylabel("Model / Geant4")
        ax[0].legend(loc="best", frameon=False, fontsize=13)
        cms_label(ax[0])

        ratio_min = min(ratio_min, 0.9)
        ratio_max = max(ratio_max, 1.1)
        ratio_pad = 0.1 * max(ratio_max - ratio_min, 0.2)
        ax[1].set_ylim(max(0.0, ratio_min - ratio_pad), ratio_max + ratio_pad)

        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
        output_base = os.path.join(output_dir, f"summary_{feature_name}")
        fig.savefig(output_base + ".png")
        fig.savefig(output_base + ".pdf", dpi=300, format="pdf")
        plt.close(fig)

    print(f"Summary plots saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay model histogram ratios (model/Geant4) from hgcal_metrics .npz outputs."
    )
    parser.add_argument(
        "--input_files",
        required=True,
        help=(
            "Inputs formatted as name1:path1,name2:path2,... "
            "where each path can be a directory of hist npz files, "
            "a single hist npz file, or a text file with one npz path per line."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="plots/summary/",
        help="Folder where summary plots are written.",
    )
    args = parser.parse_args()
    make_summary_plots(args.input_files, args.output_dir)


if __name__ == "__main__":
    main()
