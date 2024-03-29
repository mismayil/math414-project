"""
This file contains the code for training linear regression models on the generated train data for Pharmacokinetics problem.
Models are saved under data/ph/q4.
"""
import pandas as pd
import statsmodels.api as sm
import pathlib
import numpy as np

from pharmacokinetics import train_size

seed = 42

def train_model(train_df, target):
    train_x = train_df[[col for col in train_df.columns if col.startswith("x_")]]
    train_y = train_df[target]
    train_x = sm.add_constant(train_x)
    model = sm.OLS(train_y, train_x).fit()
    return model

if __name__ == '__main__':
    np.random.seed(seed)

    # Setup output directory
    data_dir = pathlib.Path("data/ph/q4")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load train data
    train_df = pd.read_csv(data_dir / f"ph_train_data_[size={train_size}].csv")
    
    # Train models
    k_a_model = train_model(train_df, target="K_a")
    k_e_model = train_model(train_df, target="K_e")
    cl_model = train_model(train_df, target="Cl")
    sigma_model = train_model(train_df, target="sigma")

    print("Model summaries:")
    print("K_a model:")
    print(k_a_model.summary())
    print("K_e model:")
    print(k_e_model.summary())
    print("Cl model:")
    print(cl_model.summary())
    print("sigma model:")
    print(sigma_model.summary())

    # Save models
    k_a_model.save(data_dir / f"ph_k_a_model_[size={train_size}].pkl")
    k_e_model.save(data_dir / f"ph_k_e_model_[size={train_size}].pkl")
    cl_model.save(data_dir / f"ph_cl_model_[size={train_size}].pkl")
    sigma_model.save(data_dir / f"ph_sigma_model_[size={train_size}].pkl")
