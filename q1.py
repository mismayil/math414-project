import numpy as np
import pathlib

from utils import Normal, run_abc_rejection, plot_samples
from example import ExamplePosteriorModel, generate_example_data, compute_example_discrepancy, example_sigma, example_M, example_mean, example_a, example_sigma_1, example_N, example_tolerances

seed = 42

if __name__ == "__main__":
    np.random.seed(seed)
    data_dir = pathlib.Path("data/ex/q1")
    data_dir.mkdir(parents=True, exist_ok=True)

    prior_model = Normal(0, example_sigma)
    example_theta = prior_model.sample()
    observed_data = generate_example_data(example_theta, example_M)
    posterior_model = ExamplePosteriorModel(example_M, example_mean, example_a, example_sigma, example_sigma_1)

    tolerance_samples = []

    for tolerance in example_tolerances:
        sample, acceptance_rate = run_abc_rejection(example_N, observed_data, prior_model, generate_example_data, compute_example_discrepancy, tolerance)
        print(f"tolerance: {tolerance}, acceptance rate: {acceptance_rate*100:.2f}%")
        tolerance_samples.append(sample)
        np.save(data_dir / f"ex_q1_[tol={tolerance}]_[N={example_N}].npy", sample)

    plot_samples(tolerance_samples, posterior_model, example_tolerances, set_log=True, set_ylim=1e-6)