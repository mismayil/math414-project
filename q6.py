"""
This file contains the code for experiments with Variance Reduction for Pharmacokinetics problem.
Generated data is saved under data/ph/q6.
"""
import pathlib
import numpy as np
import argparse
from tqdm import tqdm

from utils import run_euler_maruyama, Normal
from pharmacokinetics import PHKModel, sampling_dt, drug_dose

seed = 42

class NegativeNormal(Normal):
    """Negative normal distribution."""
    def sample(self, size=None):
        return -super().sample(size=size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=seed)
    parser.add_argument("-n", "--size", type=int, default=1000)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Setup output directory
    data_dir = pathlib.Path("data/ph/q6")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load posterior samples and compute posterior theta mean
    posterior_theta_samples = np.load("data/ph/q5/rw_lognorm/ph_rw_lognorm_[tol=0.25]_[N=10000].npy")
    posterior_theta = np.mean(posterior_theta_samples, axis=0)

    # Initialize model with posterior theta
    phk_model = PHKModel(D=drug_dose, K_a=posterior_theta[0], K_e=posterior_theta[1], Cl=posterior_theta[2], sigma=posterior_theta[3], dt=sampling_dt)
    
    # Run Crude Monte Carlo Estimation
    cmc_sample = []

    for i in tqdm(range(args.size), desc="Generating CMC samples"):
        data = run_euler_maruyama([12], phk_model, dt=sampling_dt)
        cmc_sample.append(data[0])

    print("Crude Monte Carlo Estimation of E[X_9]:")
    print(f"mean: {np.mean(cmc_sample)}, variance: {np.var(cmc_sample)}")
    np.save(data_dir / f"ph_x9_cmc_data_[size={args.size}].npy", cmc_sample)

    # Run Monte Carlo Estimation using anthithetic variates
    av_sample = []

    # This is identical to the phk_model, except that the brownian motion is negative normal
    negative_phk_model = PHKModel(D=drug_dose, K_a=posterior_theta[0], K_e=posterior_theta[1], Cl=posterior_theta[2], sigma=posterior_theta[3],
                                  dt=sampling_dt, brownian=NegativeNormal(0, np.sqrt(sampling_dt)))

    for i in tqdm(range(int(args.size/2)), desc="Generating AV samples"):
        positive_data = run_euler_maruyama([12], phk_model, dt=sampling_dt)
        negative_data = run_euler_maruyama([12], negative_phk_model, dt=sampling_dt)
        av_sample.append((positive_data[0] + negative_data[0])/2)
    
    print("AV Monte Carlo Estimation of E[X_9]:")
    print(f"mean: {np.mean(av_sample)}, variance: {np.var(av_sample)}")
    np.save(data_dir / f"ph_x9_av_data_[size={int(args.size/2)}].npy", av_sample)