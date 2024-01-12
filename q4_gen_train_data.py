from tqdm import tqdm
import pandas as pd
import pathlib
"""
This file contains the code for generating training data to estimate the model parameters for Pharmacokinetics problem.
Generated data is saved under data/ph/q4.
"""
import numpy as np

from utils import run_euler_maruyama
from pharmacokinetics import PHKModel, PHKPriorModel, sampling_dt, sampling_times, train_size, drug_dose

seed = 42

def generate_train_data(train_size=100):
    train_data = []
    prior_model = PHKPriorModel()

    for i in tqdm(range(train_size), total=train_size, desc="Generating train data"):
        prior_sample = prior_model.sample()
        priors = {
            "K_a": prior_sample[0],
            "K_e": prior_sample[1],
            "Cl": prior_sample[2],
            "sigma": prior_sample[3]
        }
        phk_model = PHKModel(D=drug_dose, dt=sampling_dt, **priors)
        sample = run_euler_maruyama(sampling_times, phk_model, dt=sampling_dt)
        train_data.append(sample + [priors["K_a"], priors["K_e"], priors["Cl"], priors["sigma"]])

    train_df = pd.DataFrame(train_data, columns=[f"x_[t={t}]" for t in sampling_times] + ["K_a", "K_e", "Cl", "sigma"])

    return train_df

if __name__ == '__main__':
    np.random.seed(seed)

    # Setup output directory
    data_dir = pathlib.Path("data/ph/q4")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate train data and save as csv
    train_df = generate_train_data(train_size=train_size)
    train_df.to_csv(data_dir / f"ph_train_data_[size={train_size}].csv", index=False)
