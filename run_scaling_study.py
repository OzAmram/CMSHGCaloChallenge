#!/usr/bin/env python3
"""Master script: run KS scaling study + produce plots for all datasets.

Runs run_ks_scaling.py then plotting/plot_ks_scaling.py for each dataset.

Usage:
    python3 run_scaling_study.py                    # all datasets
    python3 run_scaling_study.py --datasets Photon_E5 Pion_E50
    python3 run_scaling_study.py --plot_only        # skip scaling runs, just plot
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent
EOS  = "/eos/cms/store/group/offcomp-sim/HGCal_Sim_Samples_2024"

DATASETS = {
    "Photon_E5": dict(
        config    = str(REPO / "configs/config_HGCal_photons_E5.json"),
        data_dir  = f"{EOS}/SinglePhoton_E-5_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        train_dir = f"{EOS}/SinglePhoton_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        generated = str(REPO / "datasets/generated/Geant4_Photon_E5.txt"),
        train_list= str(REPO / "datasets/photon_files_train.txt"),
        energy    = 5.0,
        windows   = [0.006],
        single_energy = True,
    ),
    "Photon_E50": dict(
        config    = str(REPO / "configs/config_HGCal_photons_E50.json"),
        data_dir  = f"{EOS}/SinglePhoton_E-50_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        train_dir = f"{EOS}/SinglePhoton_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        generated = str(REPO / "datasets/generated/Geant4_Photon_E50.txt"),
        train_list= str(REPO / "datasets/photon_files_train.txt"),
        energy    = 50.0,
        windows   = [0.006],
        single_energy = True,
    ),
    "Photon_E500": dict(
        config    = str(REPO / "configs/config_HGCal_photons_E500.json"),
        data_dir  = f"{EOS}/SinglePhoton_E-500_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        train_dir = f"{EOS}/SinglePhoton_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        generated = str(REPO / "datasets/generated/Geant4_Photon_E500.txt"),
        train_list= str(REPO / "datasets/photon_files_train.txt"),
        energy    = 500.0,
        windows   = [0.006],
        single_energy = True,
    ),
    "Photon_LogUniform": dict(
        config    = str(REPO / "configs/config_HGCal_photons.json"),
        data_dir  = f"{EOS}/SinglePhoton_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        train_dir = f"{EOS}/SinglePhoton_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        generated = str(REPO / "datasets/generated/Geant4_Photon_LogUniform.txt"),
        train_list= str(REPO / "datasets/photon_files_train.txt"),
        energy    = None,
        windows   = None,
        single_energy = False,
    ),
    "Pion_E5": dict(
        config    = str(REPO / "configs/config_HGCal_pions_E5.json"),
        data_dir  = f"{EOS}/SinglePion_E-5_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        train_dir = f"{EOS}/SinglePion_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        generated = str(REPO / "datasets/generated/Geant4_Pion_E5.txt"),
        train_list= str(REPO / "datasets/pion_files_train.txt"),
        energy    = 5.0,
        windows   = [0.05],
        single_energy = True,
    ),
    "Pion_E50": dict(
        config    = str(REPO / "configs/config_HGCal_pions_E50.json"),
        data_dir  = f"{EOS}/SinglePion_E-50_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        train_dir = f"{EOS}/SinglePion_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        generated = str(REPO / "datasets/generated/Geant4_Pion_E50.txt"),
        train_list= str(REPO / "datasets/pion_files_train.txt"),
        energy    = 50.0,
        windows   = [0.05],
        single_energy = True,
    ),
    "Pion_E500": dict(
        config    = str(REPO / "configs/config_HGCal_pions_E500.json"),
        data_dir  = f"{EOS}/SinglePion_E-500_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        train_dir = f"{EOS}/SinglePion_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        generated = str(REPO / "datasets/generated/Geant4_Pion_E500.txt"),
        train_list= str(REPO / "datasets/pion_files_train.txt"),
        energy    = 500.0,
        windows   = [0.05],
        single_energy = True,
    ),
    "Pion_LogUniform": dict(
        config    = str(REPO / "configs/config_HGCal_pions.json"),
        data_dir  = f"{EOS}/SinglePion_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        train_dir = f"{EOS}/SinglePion_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s",
        generated = str(REPO / "datasets/generated/Geant4_Pion_LogUniform.txt"),
        train_list= str(REPO / "datasets/pion_files_train.txt"),
        energy    = None,
        windows   = None,
        single_energy = False,
    ),
}


def run(cmd: list, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        print(f"WARNING: command exited with code {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()),
                        help="Datasets to process (default: all)")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[10, 50] + list(range(100, 1000, 100)) + list(range(1000, 10500, 500)) + list(range(10000, 110000, 10000)),
                        help="Sample sizes for scaling study (default: 1000 3000 ... 29000)")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip the scaling runs, only produce plots")
    parser.add_argument("--xmax", type=float, default=None,
                        help="x-axis max for plots")
    parser.add_argument("--logx", action="store_true",
                        help="Use logarithmic x-axis scale in plots")
    parser.add_argument("--logy", action="store_true",
                        help="Use logarithmic y-axis scale in plots")
    parser.add_argument("--plot_dir", default=str(REPO / "eval_scaling" / "plots"),
                        help="Output directory for plots")
    flags = parser.parse_args()

    plot_dir = Path(flags.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for label in flags.datasets:
        ds = DATASETS[label]

        # --- 1. Run scaling study ---
        if not flags.plot_only:
            cmd = [
                sys.executable, str(REPO / "run_ks_scaling.py"),
                "--label",     label,
                "--config",    ds["config"],
                "--data_dir",  ds["data_dir"],
                "--generated", ds["generated"],
                "--sizes",     *[str(s) for s in flags.sizes],
            ]
            if ds["single_energy"]:
                cmd.append("--single_energy")
            run(cmd, f"Scaling study: {label}")

        # --- 2. Plot ---
        cmd = [
            sys.executable, str(REPO / "plotting/plot_ks_scaling.py"),
            "--pattern", f"{label}_n*",
            "--dataset", label,
            "-o",        str(plot_dir / f"scaling_{label}"),
        ]
        if ds["energy"] is not None:
            cmd += ["-e", str(ds["energy"]),
                    "-f", ds["train_list"],
                    "--train_dir", ds["train_dir"],
                    "--windows", *[str(w) for w in ds["windows"]]]
        if flags.xmax is not None:
            cmd += ["--xmax", str(flags.xmax)]
        if flags.logx:
            cmd.append("--logx")
        if flags.logy:
            cmd.append("--logy")
        run(cmd, f"Plotting: {label}")

    print(f"\nDone. Plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
