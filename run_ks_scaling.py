#!/usr/bin/env python3
"""Run Geant4-vs-Geant4 KS scaling study for a given dataset.

Loops over generated sample sizes, runs hgcal_metrics.py in hist mode,
then collects results into eval_scaling/ks_scaling_<LABEL>.csv.

Example:
    python3 run_ks_scaling.py \\
        --label Photon_E5 \\
        --config config_HGCal_photons_E5.json \\
        --data_dir /eos/.../h5s \\
        --generated datasets/generated/Geant4_Photon_E5.txt \\
        --single_energy
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

REPO   = Path(__file__).parent
SCRIPT = REPO / "hgcal_metrics.py"


def parse_metrics(metrics_path: Path) -> dict:
    result = {}
    for line in metrics_path.read_text().splitlines():
        m = re.match(r"Num generated showers:\s*(\d+)", line)
        if m:
            result["n_showers"] = int(m.group(1))
        m = re.match(r"Avg separation power / KS of (\w+) features:\s*([\d.e+\-]+)\s*/\s*([\d.e+\-]+)", line)
        if m:
            cat = m.group(1)
            result[f"Sep_{cat}"] = float(m.group(2))
            result[f"KS_{cat}"]  = float(m.group(3))
    return result


def run_one(n: int, out_base: Path, label: str, config: str,
            data_dir: str, generated: str, single_energy: bool) -> Path:
    out_dir = out_base / f"{label}_n{n}"
    metrics = out_dir / "metrics.txt"
    if metrics.exists():
        print(f"  [skip] n={n:>6,}  (already done)")
        return metrics

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPT),
        "--config",       config,
        "-d",             data_dir,
        "-g",             generated,
        "-p",             str(out_dir) + "/",
        "--name",         "Geant4",
        "--mode",         "hist",
        "--nevts_sample", str(n),
    ]
    if single_energy:
        cmd.append("--single_energy")

    print(f"  [run]  n={n:>6,}  -> {out_dir.name}")
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        print(f"  WARNING: exit code {result.returncode} for n={n}")
    return metrics


def collect_csv(sizes: list, out_base: Path, label: str, csv_out: Path):
    cats = ["Energy", "Transverse", "Center", "Width", "Occupancy", "all"]
    fieldnames = ["n_generated"] + [f"KS_{c}" for c in cats] + [f"Sep_{c}" for c in cats]
    rows = []
    for n in sizes:
        metrics_path = out_base / f"{label}_n{n}" / "metrics.txt"
        if not metrics_path.exists():
            print(f"  WARNING: no metrics for n={n}, skipping from CSV")
            continue
        m = parse_metrics(metrics_path)
        row = {"n_generated": m.get("n_showers", n)}
        for c in cats:
            row[f"KS_{c}"]  = m.get(f"KS_{c}",  "")
            row[f"Sep_{c}"] = m.get(f"Sep_{c}", "")
        rows.append(row)

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {csv_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",         required=True,
                        help="Dataset label used for output dir names and CSV (e.g. Photon_E5)")
    parser.add_argument("--config",        required=True,
                        help="Path to dataset JSON config")
    parser.add_argument("--data_dir",      required=True,
                        help="EOS directory containing reference h5 files")
    parser.add_argument("--generated",     required=True,
                        help="Path to generated file list (.txt)")
    parser.add_argument("--single_energy", action="store_true",
                        help="Pass --single_energy flag to hgcal_metrics.py")
    parser.add_argument("--out_base",      default=str(REPO / "eval_scaling"),
                        help="Base output directory (default: eval_scaling/)")
    parser.add_argument("--sizes",         type=int, nargs="+",
                        default=list(range(100, 1000, 100)) + list(range(1000, 11000, 500)),
                        help="Sample sizes to evaluate (default: 100 to 10k)")
    flags = parser.parse_args()

    out_base = Path(flags.out_base)
    csv_out  = out_base / f"ks_scaling_{flags.label}.csv"

    print(f"KS scaling study: {flags.label}")
    print(f"Sizes: {flags.sizes}\n")

    for n in flags.sizes:
        run_one(n, out_base, flags.label, flags.config,
                flags.data_dir, flags.generated, flags.single_energy)

    print("\nCollecting results...")
    collect_csv(flags.sizes, out_base, flags.label, csv_out)


if __name__ == "__main__":
    main()
