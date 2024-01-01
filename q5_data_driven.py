import numpy as np
import pickle
from functools import partial
import pathlib
import json

from utils import run_abc_mcmc, run_euler_maruyama
from pharmacokinetics import make_phk_data_driven_proposal_model, compute_phk_discrepancy, generate_phk_data, PHKPriorModel, train_size, PHKModel, sampling_dt, sampling_times, drug_dose

seed = 42

if __name__ == "__main__":
    np.random.seed(seed)
    data_dir = pathlib.Path("data/ph/q5/rw_dd")
    data_dir.mkdir(parents=True, exist_ok=True)
    phk_tolerances = [0.25, 0.7, 1]
    phk_N = 10000
    theta_0 = [1.15, 0.07, 0.05, 0.33]
    phk_samples = []
    burn_in = 0.1

    observed_data = np.load("data/ph/q4/ph_observed_data.npy")
    
    k_a_model = pickle.load(open(f"data/ph/q4/ph_k_a_model_[size={train_size}].pkl", "rb"))
    k_e_model = pickle.load(open(f"data/ph/q4/ph_k_e_model_[size={train_size}].pkl", "rb"))
    cl_model = pickle.load(open(f"data/ph/q4/ph_cl_model_[size={train_size}].pkl", "rb"))
    sigma_model = pickle.load(open(f"data/ph/q4/ph_sigma_model_[size={train_size}].pkl", "rb"))
    coefficients = np.vstack([k_a_model.params, k_e_model.params, cl_model.params, sigma_model.params])
    
    phk_prior_model = PHKPriorModel()
    phk_model = PHKModel(D=drug_dose, K_a=theta_0[0], K_e=theta_0[1], Cl=theta_0[2], sigma=theta_0[3], dt=sampling_dt)
    data_0 = run_euler_maruyama(sampling_times, phk_model, dt=sampling_dt, debug=True)

    # Run ABC-MCMC with data driven proposal
    for tolerance in phk_tolerances:
        sample, acceptance_rate = run_abc_mcmc(phk_N, 
                                                observed_data, 
                                                partial(make_phk_data_driven_proposal_model, coefficients=coefficients), 
                                                phk_prior_model, 
                                                generate_phk_data, 
                                                partial(compute_phk_discrepancy, coefficients, theta_0), 
                                                tolerance,
                                                theta_0=theta_0,
                                                burn_in=burn_in,
                                                data_0=data_0)
        print(f"tolerance: {tolerance}, acceptance rate: {acceptance_rate*100:.2f}%")
        phk_samples.append(sample)
        file_stem = f"ph_dd_[tol={tolerance}]_[N={phk_N}]"
        np.save(data_dir / f"{file_stem}.npy", sample)
        metadata = {
            "tolerance": tolerance,
            "phk_N": phk_N,
            "burn_in": burn_in,
            "acceptance_rate": acceptance_rate
        }
        with open(data_dir / f"{file_stem}.json", "w") as f:
            json.dump(metadata, f, indent=4)