#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# Import OpenSim
try:
    import opensim
except ImportError:
    print("Error: opensim module not found. Ensure you are in the correct conda environment.")
    sys.exit(1)

# Import from scripts/run_exo_record_dynamics.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from run_exo_record_dynamics import plate_wrench, lab_to_osim, load_markers
except ImportError as e:
    print(f"Error importing from run_exo_record_dynamics: {e}")
    sys.exit(1)

def get_com_trajectory(model_path, ik_mot_path):
    model = opensim.Model(str(model_path))
    state = model.initSystem()
    
    # Parse mot file
    lines = Path(ik_mot_path).read_text().splitlines()
    endheader_idx = lines.index("endheader")
    labels = lines[endheader_idx + 1].split()
    data = np.loadtxt(lines[endheader_idx + 2:])
    
    times = data[:, 0]
    coords_data = data[:, 1:]
    
    # Map labels to coordinates
    coord_set = model.getCoordinateSet()
    active_coords = []
    for label in labels[1:]:
        try:
            c = coord_set.get(label)
            active_coords.append((c, label))
        except Exception:
            continue
            
    com_traj = []
    for frame_idx in range(len(times)):
        for c, label in active_coords:
            # Find index of label in labels[1:]
            col_idx = labels.index(label) - 1
            val = coords_data[frame_idx, col_idx]
            if c.getMotionType() == opensim.Coordinate.Rotational:
                val = np.deg2rad(val)
            c.setValue(state, val, False)
        
        model.realizePosition(state)
        com = model.calcMassCenterPosition(state)
        com_traj.append([com.get(0), com.get(1), com.get(2)])
        
    return times, np.array(com_traj)

def lowpass_filter(data, fs, cutoff=6.0, order=4):
    sos = butter(order, cutoff, fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/exo_improve/sync.md")
    args = parser.parse_args()
    
    model_path = Path("~/nas/MAEVE/EXO_RECORD/s5/s5_opensim/calibration/model_opensim.osim").expanduser()
    output_md_path = Path(args.output)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    # Model mass
    model = opensim.Model(str(model_path))
    total_mass = sum(body.getMass() for body in model.getBodySet())
    
    results = []
    
    for trial in args.trials:
        print(f"Processing {trial}...")
        
        # Determine IK path
        if trial in ["static_0", "static_1", "posterior_reaching_end", "posterior_reaching_1"]: # Correction based on objective
            # Note: Objective says posterior_reaching_1
            ik_path = Path(f"outputs/exo_record_ownik/s5/{trial}/{trial}_ik.mot")
        elif trial in ["axial_twist_0", "axial_twist_1"]:
            ik_path = Path(f"outputs/exo_record_ownik_notorso/s5/{trial}/{trial}_ik.mot")
        else:
            # Fallback or check trial list
            ik_path = Path(f"outputs/exo_record_ownik/s5/{trial}/{trial}_ik.mot")

        if not ik_path.exists():
            # Try other path
            ik_path = Path(f"outputs/exo_record_ownik_notorso/s5/{trial}/{trial}_ik.mot")
            if not ik_path.exists():
                print(f"Warning: IK file not found for {trial}. Skipping.")
                continue

        # COM and Required Force
        times, com_pos = get_com_trajectory(model_path, ik_path)
        fs = 1.0 / np.median(np.diff(times))
        
        com_filt = lowpass_filter(com_pos, fs)
        # Acceleration: v = grad(p)*fs, a = grad(v)*fs
        vel = np.gradient(com_filt, axis=0) * fs
        acc = np.gradient(vel, axis=0) * fs
        
        # F_req = m * (a_com + [0, 9.81, 0])
        f_req = total_mass * (acc + np.array([0, 9.81, 0]))
        
        # Plate data
        plate_csv = Path(f"~/nas/MAEVE/EXO_RECORD/s5/aligned/forceplate/{trial}.csv").expanduser()
        marker_csv = Path(f"~/nas/MAEVE/EXO_RECORD/s5/aligned/marker/{trial}.csv").expanduser()
        
        # To avoid reading plate_wrench 101 times, we can simulate it by reading once 
        # and interpolating. But users said calling it 101 times is acceptable.
        # However, let's optimize slightly if possible.
        
        lags = np.arange(-0.100, 0.101, 0.002)
        rms_total = []
        rms_axes = [[], [], []] # X, Y, Z
        
        best_lag = 0
        min_rms = float('inf')
        
        # For plot
        plot_lags = []
        plot_rms = []
        
        # We'll do a loop over lags
        for lag in lags:
            shifted_times = times + lag
            # Plate wrench returns (force, cop, free_moment)
            # force is reaction ON THE SUBJECT in LAB frame (Z up)
            f_plate_lab, _, _ = plate_wrench(plate_csv, shifted_times, cutoff_hz=6.0)
            f_plate_osim = lab_to_osim(f_plate_lab)
            
            # RMS error per axis
            errs = f_plate_osim - f_req
            rms_x = np.sqrt(np.mean(errs[:, 0]**2))
            rms_y = np.sqrt(np.mean(errs[:, 1]**2))
            rms_z = np.sqrt(np.mean(errs[:, 2]**2))
            rms_tot = np.sqrt(np.mean(errs**2))
            
            rms_axes[0].append(rms_x)
            rms_axes[1].append(rms_y)
            rms_axes[2].append(rms_z)
            rms_total.append(rms_tot)
            
            if rms_tot < min_rms:
                min_rms = rms_tot
                best_lag = lag
            
            plot_lags.append(lag)
            plot_rms.append(rms_tot)

        # Metrics at lag 0
        lag_0_idx = np.argmin(np.abs(lags))
        rms_0 = rms_total[lag_0_idx]
        
        # Best lag per axis
        best_lag_x = lags[np.argmin(rms_axes[0])]
        best_lag_y = lags[np.argmin(rms_axes[1])]
        best_lag_z = lags[np.argmin(rms_axes[2])]
        
        # Mean vertical force / weight
        # Use best lag for this or lag 0? Usually best lag.
        best_lag_idx = np.argmin(rms_total)
        # Need f_plate for best lag
        f_plate_best_lab, _, _ = plate_wrench(plate_csv, times + best_lag, cutoff_hz=6.0)
        f_plate_best_osim = lab_to_osim(f_plate_best_lab)
        mean_f_y = np.mean(f_plate_best_osim[:, 1])
        weight = total_mass * 9.81
        ratio_weight = mean_f_y / weight
        
        results.append({
            "trial": trial,
            "best_lag": best_lag,
            "rms_0": rms_0,
            "rms_best": min_rms,
            "best_lag_x": best_lag_x,
            "best_lag_y": best_lag_y,
            "best_lag_z": best_lag_z,
            "ratio_weight": ratio_weight,
            "plot_data": (plot_lags, plot_rms)
        })
        
        if trial == "axial_twist_0":
            plt.figure(figsize=(8, 5))
            plt.plot(plot_lags, plot_rms)
            plt.xlabel("Lag (s)")
            plt.ylabel("Total RMS (N)")
            plt.title(f"Sync Check: {trial}")
            plt.grid(True)
            plt.savefig("outputs/exo_improve/sync_axial_twist_0.png")
            plt.close()

    # Write MD report
    with open(output_md_path, "w") as f:
        f.write("# Sync Check Report\n\n")
        for res in results:
            f.write(f"## Trial: {res['trial']}\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            f.write(f"| Best Lag (total) | {res['best_lag']:.4f} s |\n")
            f.write(f"| RMS at Lag 0 | {res['rms_0']:.4f} N |\n")
            f.write(f"| RMS at Best Lag | {res['rms_best']:.4f} N |\n")
            f.write(f"| Best Lag X | {res['best_lag_x']:.4f} s |\n")
            f.write(f"| Best Lag Y | {res['best_lag_y']:.4f} s |\n")
            f.write(f"| Best Lag Z | {res['best_lag_z']:.4f} s |\n")
            f.write(f"| Mean F_y / Weight | {res['ratio_weight']:.4f} |\n\n")
            
            # Comments
            if "static" in res['trial']:
                f.write("Comment: Static trial. Signal is very low, best lag is not meaningful.\n\n")
            else:
                f.write(f"Comment: Dynamic trial. Best lag is {res['best_lag']:.4f}s.\n\n")

if __name__ == "__main__":
    main()
