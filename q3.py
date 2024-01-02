from functools import partial
import numpy as np
import pathlib
import json

from utils import run_abc_mcmc, Normal, plot_samples
from example import example_tolerances, make_example_proposal_model, generate_example_data, compute_example_discrepancy, example_sigma, example_M, example_N, ExamplePosteriorModel, example_mean, example_a, example_sigma_1

seed = 42

if __name__ == "__main__":
    np.random.seed(42)
    data_dir = pathlib.Path("data/ex/q3")
    data_dir.mkdir(parents=True, exist_ok=True)

    prior_model = Normal(0, example_sigma)
    example_theta = prior_model.sample()
    observed_data = generate_example_data(example_theta, example_M)
    posterior_model = ExamplePosteriorModel(example_M, example_mean, example_a, example_sigma, example_sigma_1)

    proposal_sigmas = [0.01, 0.1, 0.5, 1, 3]

    for proposal_sigma in proposal_sigmas:
        tolerance_samples = []
        
        for tolerance in example_tolerances:
            sample, acceptance_rate = run_abc_mcmc(example_N, observed_data, partial(make_example_proposal_model, sigma=proposal_sigma), prior_model, generate_example_data, compute_example_discrepancy, tolerance)
            print(f"tolerance: {tolerance}, acceptance rate: {acceptance_rate*100:.2f}%")
            tolerance_samples.append(sample)
            file_stem = f"ex_q3_[psigma={proposal_sigma}]_[tol={tolerance}]_[N={example_N}]"
            np.save(data_dir / f"{file_stem}.npy", sample)
            metadata = {
                "seed": seed,
                "proposal_sigma": proposal_sigma,
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
    
        plot_samples(tolerance_samples, posterior_model, example_tolerances, set_log=True, set_ylim=1e-6)