from tqdm import tqdm
import pandas as pd
import statsmodels.api as sm
import pathlib
import numpy as np

from utils import run_euler_maruyama, LogNormal
from pharmacokinetics import PharmacokineticModel, sampling_dt, sampling_times, train_size

seed = 42

def generate_train_data(train_size=100):
    prior_K_a_model = LogNormal(0.14, 0.4)
    prior_K_e_model = LogNormal(-2.7, 0.6)
    prior_Cl_model = LogNormal(-3, 0.8)
    prior_sigma_model = LogNormal(-1.1, 0.3)

    train_data = []

    for i in tqdm(range(train_size), total=train_size, desc="Generating train data"):
        priors = {
            "K_a": prior_K_a_model.sample(),
            "K_e": prior_K_e_model.sample(),
            "Cl": prior_Cl_model.sample(),
            "sigma": prior_sigma_model.sample()
        }
        pharmacokinetic_model = PharmacokineticModel(D=4, dt=sampling_dt, **priors)
        sample = run_euler_maruyama(sampling_times, pharmacokinetic_model, dt=sampling_dt)
        train_data.append(sample + [priors["K_a"], priors["K_e"], priors["Cl"], priors["sigma"]])

    train_df = pd.DataFrame(train_data, columns=[f"x_[t={t}]" for t in sampling_times] + ["K_a", "K_e", "Cl", "sigma"])

    return train_df

if __name__ == '__main__':
    np.random.seed(seed)
    data_dir = pathlib.Path("data/ph/q4")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_df = generate_train_data(train_size=train_size)
    train_df.to_csv(data_dir / f"ph_train_data_[size={train_size}].csv", index=False)
