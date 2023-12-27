import numpy as np
import pickle
from functools import partial
import pathlib

from utils import run_abc_mcmc
from pharmacokinetics import make_pharmacokinetic_proposal_model, compute_pharmacokinetics_discrepancy, generate_pharmacokinetics_data, PharmacokineticPriorModel, train_size

SEED = 42

if __name__ == "__main__":
    np.random.seed(SEED)
    data_dir = pathlib.Path("data/ph/q5")
    data_dir.mkdir(parents=True, exist_ok=True)
    pharmacokinetics_tolerances = [0.25, 0.7, 1]
    pharmacokinetics_N = 100
    theta_0 = [1.15, 0.07, 0.05, 0.33]
    pharmacokinetics_samples = []

    observed_data = np.load("data/ph/q4/ph_observed_data.npy")
    
    k_a_model = pickle.load(open(f"data/ph/q4/ph_k_a_model_[size={train_size}].pkl", "rb"))
    k_e_model = pickle.load(open(f"data/ph/q4/ph_k_e_model_[size={train_size}].pkl", "rb"))
    cl_model = pickle.load(open(f"data/ph/q4/ph_cl_model_[size={train_size}].pkl", "rb"))
    sigma_model = pickle.load(open(f"data/ph/q4/ph_sigma_model_[size={train_size}].pkl", "rb"))
    coefficients = np.vstack([k_a_model.params, k_e_model.params, cl_model.params, sigma_model.params])
    
    pharmacokinetic_prior_model = PharmacokineticPriorModel()

    for tolerance in pharmacokinetics_tolerances:
        sample, acceptance_rate = run_abc_mcmc(pharmacokinetics_N, 
                                                observed_data, 
                                                make_pharmacokinetic_proposal_model, 
                                                pharmacokinetic_prior_model, 
                                                generate_pharmacokinetics_data, 
                                                partial(compute_pharmacokinetics_discrepancy, coefficients, theta_0), 
                                                tolerance,
                                                theta_0=theta_0)
        print(f"tolerance: {tolerance}, acceptance rate: {acceptance_rate*100:.2f}%")
        pharmacokinetics_samples.append(sample)
        np.save(data_dir / f"ph_[tol={tolerance}]_[N={pharmacokinetics_N}].npy", sample)