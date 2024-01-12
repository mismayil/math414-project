"""
This file contains the code for experiments with ABC Rejection algorithm on the synthetic (example) problem.
Experiment results are saved in data/ex/q1.
"""
import numpy as np
import pathlib
import json

from utils import Normal, run_abc_rejection
from example import generate_example_data, compute_example_discrepancy, example_sigma, example_M, example_mean, example_a, example_sigma_1, example_N, example_tolerances

seed = 42

if __name__ == "__main__":
    np.random.seed(seed)
    data_dir = pathlib.Path("data/ex/q1")
    data_dir.mkdir(parents=True, exist_ok=True)

    prior_model = Normal(0, example_sigma)
    
    # Here although we know the mean of the observed data, we generate one to be able to pass to the interface
    # discrepancy function ignores this data and assumes mean 0
    example_theta = prior_model.sample()
    observed_data = generate_example_data(example_theta, example_M)

    tolerance_samples = []

    for tolerance in example_tolerances:
        sample, acceptance_rate = run_abc_rejection(example_N, observed_data, prior_model, generate_example_data, compute_example_discrepancy, tolerance)
        print(f"tolerance: {tolerance}, acceptance rate: {acceptance_rate*100:.2f}%")
        tolerance_samples.append(sample)

        # Save data
        file_stem = f"ex_q1_[tol={tolerance}]_[N={example_N}]"
        np.save(data_dir / f"{file_stem}.npy", sample)
        metadata = {
            "seed": seed,
            "tolerance": tolerance,
            "N": example_N,
            "M": example_M,
            "a": example_a,
            "sigma": example_sigma,
            "sigma_1": example_sigma_1,
            "data_mean": example_mean,
            "acceptance_rate": acceptance_rate
        }
        with open(data_dir / f"{file_stem}.json", "w") as f:
            json.dump(metadata, f, indent=4)