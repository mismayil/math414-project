"""
This file contains the code for generating observed data for Pharmacokinetics problem.
Generated data is saved under data/ph/q4.
"""
import pathlib
import numpy as np

from utils import run_euler_maruyama
from pharmacokinetics import PHKModel, sampling_dt, sampling_times, drug_dose

seed = 42

if __name__ == '__main__':
    np.random.seed(seed)

    # Setup output directory
    data_dir = pathlib.Path("data/ph/q4")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate observed data and save as npy
    phk_model = PHKModel(D=drug_dose, K_a=1.5, K_e=0.08, Cl=0.04, sigma=0.2, dt=sampling_dt)
    observed_data = run_euler_maruyama(sampling_times, phk_model, dt=sampling_dt, debug=True)

    np.save(data_dir / f"ph_observed_data.npy", observed_data)
