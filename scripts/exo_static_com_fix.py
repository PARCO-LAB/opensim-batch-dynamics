import os
import sys
from pathlib import Path
import numpy as np
import opensim

# Add scripts directory to path so we can import helper functions
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_exo_record_dynamics import lab_to_osim, plate_wrench

def get_com_from_ik(model, ik_mot_path):
    """Compute the COM for each frame of a .mot file."""
    state = model.initSystem()
    
    # Read .mot file
    lines = ik_mot_path.read_text().splitlines()
    endheader = lines.index("endheader")
    labels = lines[endheader + 1].split()
    values = np.loadtxt(lines[endheader + 2 :], ndmin=2)
    
    column = {name: i for i, name in enumerate(labels)}
    coords = model.getCoordinateSet()
    
    coms = []
    for row in values:
        for i in range(coords.getSize()):
            coord = coords.get(i)
            name = coord.getName()
            if name in column:
                val = row[column[name]]
                # Assume .mot is in degrees for rotational coordinates
                if coord.getMotionType() == opensim.Coordinate.Rotational:
                    val = np.deg2rad(val)
                coord.setValue(state, val, False)
        
        model.realizePosition(state)
        com = model.calcMassCenterPosition(state)
        coms.append([com.get(0), com.get(1), com.get(2)])
        
    return np.array(coms)

def main():
    # Paths
    subject_root = Path("~/nas/MAEVE/EXO_RECORD/s5").expanduser()
    model_path = subject_root / "s5_opensim/calibration/model_opensim.osim"
    ik_root = Path("outputs/exo_record_ownik/s5")
    
    model = opensim.Model(str(model_path))
    state = model.initSystem()
    model_mass = model.getTotalMass(state)
    
    statics = ["static_0", "static_1"]
    all_deltas = []
    
    print(f"Model Total Mass: {model_mass:.4f} kg")
    
    for trial in statics:
        ik_mot = ik_root / trial / f"{trial}_ik.mot"
        plate_csv = subject_root / "aligned/forceplate" / f"{trial}.csv"
        
        # Get COM from IK
        coms = get_com_from_ik(model, ik_mot)
        
        # Get CoP from plate
        # Plate wrench needs times. We'll extract them from the .mot
        lines = ik_mot.read_text().splitlines()
        endheader = lines.index("endheader")
        values = np.loadtxt(lines[endheader + 2 :], ndmin=2)
        times = values[:, 0]
        
        _, cops_lab, _ = plate_wrench(plate_csv, times, 6.0)
        cops = np.array([lab_to_osim(p) for p in cops_lab])
        
        # delta = CoP - COM
        delta_trial = np.mean(cops - coms, axis=0)
        print(f"Trial {trial} delta (X, Y, Z): {delta_trial}")
        all_deltas.append(delta_trial)
        
    mean_delta = np.mean(all_deltas, axis=0)
    # We only care about horizontal components X and Z in OpenSim (Y is up)
    delta_xz = np.array([mean_delta[0], 0, mean_delta[2]])
    print(f"Mean Delta XZ (m): {delta_xz}")
    
    # Find heaviest trunk body
    bodies = model.getBodySet()
    body_masses = []
    for i in range(bodies.getSize()):
        body = bodies.get(i)
        body_masses.append((body.getName(), body.getMass(), body))
    
    # Sort by mass descending
    body_masses.sort(key=lambda x: x[1], reverse=True)
    
    target_body = None
    if "thorax" in [b[0] for b in body_masses]:
        target_body = [b for b in body_masses if b[0] == "thorax"][0]
    else:
        # Heavier body above pelvis. Pelvis usually has a specific name or is the root.
        # We look for the heaviest that isn't 'pelvis' or similar.
        for name, mass, body in body_masses:
            if "pelvis" not in name.lower():
                target_body = (name, mass, body)
                print(f"Using heaviest non-pelvis body: {name} ({mass:.4f} kg)")
                break
    
    if target_body is None:
        raise RuntimeError("Could not find a suitable body to shift COM")
        
    body_name, body_mass, body_obj = target_body
    if "thorax" in body_name:
        print(f"Using thorax body ({body_mass:.4f} kg)")
    
    # Calculate local shift
    # d_local = R^T * (delta * M_total / m_body)
    
    # Get mean pose for rotation
    # We can use the mean of the first static trial as a representative pose
    state = model.initSystem()
    lines = (ik_root / statics[0] / f"{statics[0]}_ik.mot").read_text().splitlines()
    endheader = lines.index("endheader")
    values = np.loadtxt(lines[endheader + 2 :], ndmin=2)
    
    # Use the middle frame of static_0
    mid_idx = len(values) // 2
    row = values[mid_idx]
    column = {name: i for i, name in enumerate(lines[endheader + 1].split())}
    for i in range(model.getCoordinateSet().getSize()):
        coord = model.getCoordinateSet().get(i)
        name = coord.getName()
        if name in column:
            val = row[column[name]]
            if coord.getMotionType() == opensim.Coordinate.Rotational:
                val = np.deg2rad(val)
            coord.setValue(state, val, False)
    model.realizePosition(state)
    
    R = body_obj.getTransformInGround(state).R()
    # R is a Rotation matrix. In OpenSim Py, it might be a SimTK rotation.
    # Convert to numpy.
    R_np = np.array([[R.get(0,0), R.get(0,1), R.get(0,2)],
                     [R.get(1,0), R.get(1,1), R.get(1,2)],
                     [R.get(2,0), R.get(2,1), R.get(2,2)]])
    
    world_shift = delta_xz * (model_mass / body_mass)
    local_shift = R_np.T @ world_shift
    
    print(f"Local shift for {body_name} (m): {local_shift}")
    
    # Update model
    current_center = body_obj.getMassCenter()
    new_center = opensim.Vec3(
        current_center.get(0) + local_shift[0],
        current_center.get(1) + local_shift[1],
        current_center.get(2) + local_shift[2]
    )
    body_obj.setMassCenter(new_center)
    
    # Save model
    out_dir = Path("outputs/exo_record_models/s5")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_model_path = out_dir / "model_opensim_comfix.osim"
    model.printToXML(str(out_model_path))
    print(f"Saved corrected model to {out_model_path}")
    
    # Verify
    print("Verifying COM fix...")
    static_0_mot = ik_root / statics[0] / f"{statics[0]}_ik.mot"
    coms_new = get_com_from_ik(model, static_0_mot)
    
    # Extract times from the same mot file to ensure length match
    lines_v = static_0_mot.read_text().splitlines()
    endheader_v = lines_v.index("endheader")
    values_v = np.loadtxt(lines_v[endheader_v + 2 :], ndmin=2)
    times_v = values_v[:, 0]

    _, cops_lab_new, _ = plate_wrench((subject_root / "aligned/forceplate" / f"{statics[0]}.csv"), times_v, 6.0)
    cops_new = np.array([lab_to_osim(p) for p in cops_lab_new])
    
    new_delta = np.mean(cops_new - coms_new, axis=0)
    new_delta_xz = np.array([new_delta[0], 0, new_delta[2]])
    dist = np.linalg.norm(new_delta_xz)
    print(f"New mean delta XZ magnitude: {dist*1000:.4f} mm")
    if dist > 0.001:
        print("Warning: COM fix not within 1mm tolerance")
    else:
        print("COM fix verified successfully")

if __name__ == "__main__":
    main()
