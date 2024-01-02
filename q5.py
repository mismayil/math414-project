import numpy as np
import pickle
from functools import partial
import pathlib
import json
import argparse

from utils import run_abc_mcmc
from pharmacokinetics import (make_phk_rw_lognorm_proposal_model, 
                              make_phk_rw_norm_proposal_model, 
                              make_phk_rw_multi_norm_proposal_model, 
                              make_phk_rw_data_driven_proposal_model,
                              make_phk_adaptive_proposal_model, 
                              compute_phk_discrepancy, generate_phk_data, 
                              PHKPriorModel, train_size, drug_dose, sampling_dt,
                              PHKModel, sampling_times, run_euler_maruyama)

seed = 42
phk_tolerances = [0.25, 0.7, 1]
phk_N = 10000
theta_0 = [1.15, 0.07, 0.05, 0.33]
phk_samples = []
burn_in = 0.1
window_size = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=seed)
    parser.add_argument("-p", "--proposal", type=str, default="adaptive")
    parser.add_argument("-N", "--N", type=int, default=phk_N)
    parser.add_argument("-w", "--window_size", type=int, default=window_size)
    parser.add_argument("-b", "--burn_in", type=float, default=burn_in)
    parser.add_argument("-o", "--output-dir", type=str, default="data/ph/q5")

    args = parser.parse_args()

    np.random.seed(args.seed)
    data_dir = pathlib.Path(f"{args.output_dir}/{args.proposal}")
    data_dir.mkdir(parents=True, exist_ok=True)

    observed_data = np.load("data/ph/q4/ph_observed_data.npy")
    
    k_a_model = pickle.load(open(f"data/ph/q4/ph_k_a_model_[size={train_size}].pkl", "rb"))
    k_e_model = pickle.load(open(f"data/ph/q4/ph_k_e_model_[size={train_size}].pkl", "rb"))
    cl_model = pickle.load(open(f"data/ph/q4/ph_cl_model_[size={train_size}].pkl", "rb"))
    sigma_model = pickle.load(open(f"data/ph/q4/ph_sigma_model_[size={train_size}].pkl", "rb"))
    coefficients = np.vstack([k_a_model.params, k_e_model.params, cl_model.params, sigma_model.params])
    
    phk_prior_model = PHKPriorModel()

    proposal_map = {
        "rw_lognorm": make_phk_rw_lognorm_proposal_model,
        "rw_norm": make_phk_rw_norm_proposal_model,
        "rw_multinorm": make_phk_rw_multi_norm_proposal_model,
        "rw_dd": partial(make_phk_rw_data_driven_proposal_model, coefficients=coefficients),
        "adaptive": partial(make_phk_adaptive_proposal_model, t_0=int(args.N*args.burn_in), window_size=args.window_size)
    }

    data_0 = None

    if args.proposal == "rw_dd":
        phk_model = PHKModel(D=drug_dose, K_a=theta_0[0], K_e=theta_0[1], Cl=theta_0[2], sigma=theta_0[3], dt=sampling_dt)
        data_0 = run_euler_maruyama(sampling_times, phk_model, dt=sampling_dt, debug=True)

    for tolerance in phk_tolerances:
        sample, acceptance_rate = run_abc_mcmc(args.N, 
                                                observed_data, 
                                                proposal_map[args.proposal], 
                                                phk_prior_model, 
                                                generate_phk_data, 
                                                partial(compute_phk_discrepancy, coefficients, theta_0), 
                                                tolerance,
                                                theta_0=theta_0,
                                                burn_in=args.burn_in,
                                                data_0=data_0)
        print(f"tolerance: {tolerance}, acceptance rate: {acceptance_rate*100:.2f}%")
        phk_samples.append(sample)
        file_stem = f"ph_{args.proposal}_[tol={tolerance}]_[N={args.N}]"
        np.save(data_dir / f"{file_stem}.npy", sample)
        metadata = {
            "seed": args.seed,
            "tolerance": tolerance,
            "phk_N": args.N,
            "burn_in": args.burn_in,
            "acceptance_rate": acceptance_rate,
            "window_size": args.window_size,
        }
        with open(data_dir / f"{file_stem}.json", "w") as f:
            json.dump(metadata, f, indent=4)