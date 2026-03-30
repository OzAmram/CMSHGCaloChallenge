import subprocess
import time
import threading
import os
from typing import Optional, Literal, Callable, Union
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import numpy as np
import mplhep as hep


class Executable:
    def __init__(self, batch_size: int, n_samples: int, energy_range: Literal[5, 50, 500]):
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.energy_range = energy_range
        self.collection_dir = f"{Path(__file__).parent}/{self.executable_path()}"

    def executable(self):
        return NotImplementedError("Executable as a string to be passed to subprocess, using batch_size, n_samples, energy_range as class attrs")

    def container_path(self):
        return NotImplementedError("Path to the container, if applicable, used for VRAM monitoring")

    def executable_path(self):
        return NotImplementedError("Path to the base directory, used for running the executable")

    def __call__(self, ):
        command = self.executable()
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=self.collection_dir)
        if result.returncode != 0:
            print(f"Error executing command: {result.stderr}")


class TimingVRAMMonitor: 
    """
    Combined class that measures both timing and VRAM usage in a single execution.
    Runs the executable once and captures both timing and VRAM metrics simultaneously.
    """
    
    def __init__(self, test_executable: Executable, energy_range: Literal[5, 50, 500]):
        self.executable = test_executable
        self.energy_range = energy_range
        self.interval = 5  # VRAM reading interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._peak_vram = 0
        self._readings: list[int] = []
    
    def _monitor_loop(self):
        """Internal loop that continuously monitors GPU memory."""
        while self._running:
            memory_mb = self.get_gpu_memory_mb()
            if memory_mb > 0:
                self._readings.append(memory_mb)
                self._peak_vram = max(self._peak_vram, memory_mb)
            time.sleep(self.interval)
    
    def start_monitoring(self):
        """Start monitoring VRAM usage in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop_monitoring(self) -> int:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self._peak_vram
    
    def get_gpu_memory_mb(self) -> int:
        """
        Get current GPU memory usage in MB using nvidia-smi.
        Returns memory used in MB.
        """
        nvidia_run = ["apptainer", "exec", "--nv", self.executable.container_path()]
        try:
            result = subprocess.run(
                nvidia_run + ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                shell=True
            )
            if result.returncode == 0:
                # Get the first GPU's memory usage (in MB)
                memory_used = int(result.stdout.strip().split('\n')[0])
                return memory_used
        except (subprocess.SubprocessError, ValueError, IndexError):
            pass
        return -1
    
    def run(self, batch_size: Literal[1, 10, 100, 1000], n_samples: Literal[1, 5000]) -> dict:
        """
        Run the executable and measure both timing and VRAM.
        Returns a dict with timing and VRAM metrics.
        """
        # Update the executable instance with new parameters
        self.executable.batch_size = batch_size
        self.executable.n_samples = n_samples
        self.executable.energy_range = self.energy_range
        
        # Start VRAM monitoring before execution
        self.start_monitoring()
        
        # Time the execution
        start = time.time()
        self.executable()  # Call the executable instance directly
        end = time.time()
        
        # Stop VRAM monitoring after execution
        peak_vram = self.stop_monitoring()
        #peak_vram = 0 
        return {
            "time_taken": end - start,
            "peak_vram": peak_vram
        }
    
    def baseline(self) -> dict:
        "Run a baseline measure (batch_size=1, n_samples=1) for both timing and VRAM"
        return self.run(batch_size=1, n_samples=1)
    
    def __call__(self, name) -> list[dict]:
        "Run the batch tests, returning combined timing and VRAM results"
        n_samples = 2000 # Fixed samples
        results = []

        # Baseline (batch_size=1, n_samples=1)
        baseline_result = self.baseline(n_samples=5)
        results.append({
            "energy": self.energy_range,
            "batch": 1,
            "name": name,
            "time_taken": baseline_result["time_taken"],
            "peak_vram": baseline_result["peak_vram"]
        })

        # Test with larger batch sizes
        for batch in [10, 100, 1000]:
            result = self.run(batch_size=batch, n_samples=n_samples)
            results.append({
                "energy": self.energy_range,
                "batch": batch,
                "name": name,
                "time_taken": result["time_taken"],
                "peak_vram": result["peak_vram"]
            })
        
        return results

class PlotProfiling:
    def __init__(self, results_path: str = "results.json"):
        with open(results_path, "r") as f:
            self.results = json.load(f)
        # CMS CVD-friendly color palette (Petroff, adopted by CMS 2024)
        self.CMS_COLORS =["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]

        self.energies = [5, 50, 500]
        self.batch_sizes = [1, 10, 100, 1000]

        self.executables = set()
        for trial in self.results:
            for result in trial:
                self.executables.add(result["name"])


    def set_style(self): 
        """Apply CMS style using mplhep"""
        plt.style.use(hep.style.CMS)
        mpl.rcParams.update({
            "axes.labelpad": 5,
            "axes.prop_cycle": mpl.cycler(color=self.CMS_COLORS, marker=['o', 's', 'D', '^', 'v', 'P']),
            "legend.frameon": False,
            "legend.handletextpad": 0.8,
            "yaxis.labellocation": "center",
        })

    def plot_timing(self, output_dir: str = "profiling_plots"):
        self.set_style()
        for energy in self.energies:
            # --- Timing Plot (subtract baseline) ---
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
            hep.cms.label("Preliminary", ax=ax1, data=False, rlabel=f"Photon {energy} GeV", loc=0)
            
            
            for idx, exe in enumerate(self.executables):
                # Get results for this executable and energy
                energy_data = []
                for trial in self.results:
                    for result in trial:
                        if result["name"] == exe and result["energy"] == energy:
                            energy_data.append(result)
                
                if energy_data:
                    # Calculate mean time for each batch size
                    times_by_batch = {bs: [] for bs in self.batch_sizes}
                    for r in energy_data:
                        times_by_batch[r["batch"]].append(r["time_taken"])
                    
                    # Get baseline timing (batch_size=1)
                    baseline_time = np.mean(times_by_batch[1])
                    baseline_std = np.std(times_by_batch[1])
                    
                    # Subtract baseline from all timing measurements
                    mean_times = [np.mean(times_by_batch[bs]) - baseline_time for bs in self.batch_sizes]
                    
                    # Calculate error for each batch size (propagated error from baseline subtraction)
                    errors = [np.sqrt(np.std(times_by_batch[bs])**2 + baseline_std**2) for bs in self.batch_sizes]
                    
                    # Use scatter plot with error bars and different marker for each executable
                    ax1.errorbar(self.batch_sizes, mean_times, yerr=errors, markersize=8,
                            capsize=3, linestyle='none', alpha=0.7, label=exe)
                    
            ax1.set_xlabel("Batch Size")
            ax1.set_ylabel("Time/Batch (s)", loc='top')
            ax1.set_xscale('log')
            ax1.legend(loc='upper right')
            
            # Save timing plot
            save_path = f"{output_dir}/profiling_timing_{energy}GeV.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
            

    def plot_vram(self, output_dir: str = "profiling_plots"):
        self.set_style()
        for energy in self.energies:
            fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
            hep.cms.label("Preliminary", ax=ax2, data=False, rlabel=f"Photon {energy} GeV", loc=0)
            
            for exe in self.executables:
                # Get results for this executable and energy
                energy_data = []
                for trial in self.results:
                    for result in trial:
                        if result["name"] == exe and result["energy"] == energy:
                            energy_data.append(result)
                
                if energy_data:
                    # Calculate mean VRAM for each batch size (exclude baseline=1)
                    vram_by_batch = {bs: [] for bs in self.batch_sizes if bs != 1}
                    for r in energy_data:
                        if r["batch"] != 1:  # Exclude baseline
                            vram_by_batch[r["batch"]].append(r["peak_vram"])
                    
                    # Get batch sizes excluding baseline
                    vram_batch_sizes = [bs for bs in self.batch_sizes if bs != 1]
                    mean_vram = [np.mean(vram_by_batch[bs]) for bs in vram_batch_sizes]
                    std_vram = [np.std(vram_by_batch[bs]) for bs in vram_batch_sizes]
                    
                    # Use scatter plot with error bars
                    ax2.errorbar(vram_batch_sizes, mean_vram, yerr=std_vram, markersize=8,
                            capsize=3, linestyle='none', alpha=0.7, label=exe)
            
            ax2.set_xlabel("Batch Size")
            ax2.set_ylabel("Peak VRAM (MB)", loc='top')
            ax2.set_xscale('log')
            ax2.legend(loc='upper right')
            
            # Save VRAM plot
            save_path = f"{output_dir}/profiling_vram_{energy}GeV.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
