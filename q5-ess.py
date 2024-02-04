"""
This file contains the code for experiments with ABC MCMC algorithm for Pharmacokinetics problem.
Generated data is saved under data/ph/q5.
"""
import numpy as np
import pickle
from functools import partial
import pathlib
import json
import argparse

from utils import run_abc_mcmc_ess
from pharmacokinetics import (make_phk_rw_lognorm_proposal_model,  
                              make_phk_rw_data_driven_proposal_model,
                              compute_phk_discrepancy, generate_phk_data, 
                              PHKPriorModel, train_size, drug_dose, sampling_dt,
                              PHKModel, sampling_times, run_euler_maruyama, phk_tolerances)

seed = 42
phk_N = 10000
theta_0 = [1.15, 0.07, 0.05, 0.33]
burn_in = 0.1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=seed)
    parser.add_argument("-p", "--proposal", type=str, default="rw_lognorm")
    parser.add_argument("-N", "--N", type=int, default=phk_N)
    parser.add_argument("-b", "--burn_in", type=float, default=burn_in)
    parser.add_argument("-e", "--ess-lag", type=int, default=10)
    parser.add_argument("-o", "--output-dir", type=str, default="data/ph/q5-ess")

    args = parser.parse_args()

    np.random.seed(args.seed)

    # Setup output directory
    data_dir = pathlib.Path(f"{args.output_dir}/{args.proposal}")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load observed data
    observed_data = np.load("data/ph/q4/ph_observed_data.npy")
    
    # Load model coefficients
    k_a_model = pickle.load(open(f"data/ph/q4/ph_k_a_model_[size={train_size}].pkl", "rb"))
    k_e_model = pickle.load(open(f"data/ph/q4/ph_k_e_model_[size={train_size}].pkl", "rb"))
    cl_model = pickle.load(open(f"data/ph/q4/ph_cl_model_[size={train_size}].pkl", "rb"))
    sigma_model = pickle.load(open(f"data/ph/q4/ph_sigma_model_[size={train_size}].pkl", "rb"))
    coefficients = np.vstack([k_a_model.params, k_e_model.params, cl_model.params, sigma_model.params])
    
    # Initialize prior model
    phk_prior_model = PHKPriorModel()

    # Initialize proposal models
    proposal_map = {
        "rw_lognorm": make_phk_rw_lognorm_proposal_model,
        "rw_dd": partial(make_phk_rw_data_driven_proposal_model, coefficients=coefficients)
    }

    data_0 = None

    if args.proposal == "rw_dd":
        # If we are using the data-driven proposal, we need to generate data for the initial theta
        phk_model = PHKModel(D=drug_dose, K_a=theta_0[0], K_e=theta_0[1], Cl=theta_0[2], sigma=theta_0[3], dt=sampling_dt)
        data_0 = run_euler_maruyama(sampling_times, phk_model, dt=sampling_dt, debug=True)

    phk_samples = []

    for tolerance in phk_tolerances:
        past_sample = np.load(f"data/ph/q5/{args.proposal}/ph_{args.proposal}_[tol={tolerance}]_[N={args.N}].npy").tolist()
        past_data = [None]

        if args.proposal == "rw_dd":
            phk_model = PHKModel(D=drug_dose, K_a=past_sample[-1][0], K_e=past_sample[-1][1], Cl=past_sample[-1][2], sigma=past_sample[-1][3], dt=sampling_dt)
            data_0 = run_euler_maruyama(sampling_times, phk_model, dt=sampling_dt, debug=True)
            past_data = [data_0]

        sample, acceptance_rate = run_abc_mcmc_ess(past_sample, past_data, args.N, 
                                                observed_data, 
                                                proposal_map[args.proposal], 
                                                phk_prior_model, 
                                                generate_phk_data, 
                                                partial(compute_phk_discrepancy, coefficients, theta_0), 
                                                tolerance,
                                                theta_0=theta_0,
                                                burn_in=args.burn_in,
                                                ess_lag=args.ess_lag,
                                                data_0=data_0)
        print(f"tolerance: {tolerance}, acceptance rate: {acceptance_rate*100:.2f}%")
        phk_samples.append(sample)
        
        # Save data
        file_stem = f"ph_{args.proposal}_[tol={tolerance}]_[N={args.N}]"
        np.save(data_dir / f"{file_stem}.npy", sample)
        metadata = {
            "seed": args.seed,
            "tolerance": tolerance,
            "phk_N": args.N,
            "burn_in": args.burn_in,
            "ess_lag": args.ess_lag,
            "acceptance_rate": acceptance_rate,
        }
        with open(data_dir / f"{file_stem}.json", "w") as f:
            json.dump(metadata, f, indent=4)