"""
This file contains common code for the pharmacokinetics problem.
"""
from utils import Model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

from utils import Normal, LogNormal, weighted_euclidean_norm, run_euler_maruyama

# Constants for the pharmacokinetics problem
sampling_dt = 0.005
sampling_times = [0.25, 0.5, 1, 2, 3.5, 5, 7, 9, 12]
train_size = 1000
drug_dose = 4
phk_tolerances = [0.25, 0.7, 1]

class PHKModel(Model):
    """Pharmacokinetics model given as a stochastic differential equation (SDE)."""

    def __init__(self, D, K_a, K_e, Cl, sigma=1, dt=0.01, brownian=None):
        self.D = D
        self.K_a = K_a
        self.K_e = K_e
        self.Cl = Cl
        self.sigma = sigma
        self.dt = dt
        self.brownian = Normal(0, np.sqrt(dt)) if not brownian else brownian
    
    def sample(self, size=None, x_t=0, t=0.01):
        return ((self.D * self.K_a * self.K_e * np.exp(-self.K_a * t))/self.Cl - self.K_e * x_t) * self.dt + self.sigma * self.brownian.sample(size=size)

class PHKPriorModel(Model):
    """Prior model for the pharmacokinetics problem."""

    def __init__(self):
        self.prior_K_a = LogNormal(0.14, 0.4)
        self.prior_K_e = LogNormal(-2.7, 0.6)
        self.prior_Cl = LogNormal(-3, 0.8)
        self.prior_sigma = LogNormal(-1.1, 0.3)
    
    def sample(self, size=None):
        return [self.prior_K_a.sample(), self.prior_K_e.sample(), self.prior_Cl.sample(), self.prior_sigma.sample()]

    def pdf(self, x):
        return self.prior_K_a.pdf(x[0]) * self.prior_K_e.pdf(x[1]) * self.prior_Cl.pdf(x[2]) * self.prior_sigma.pdf(x[3])

class PHKRWLogNormProposalModel(Model):
    """Random walk log-normal proposal model for the pharmacokinetics problem."""

    def __init__(self, theta):
        self.k_a_model = LogNormal(np.log(theta[0]), 0.4)
        self.k_e_model = LogNormal(np.log(theta[1]), 0.6)
        self.cl_model = LogNormal(np.log(theta[2]), 0.8)
        self.sigma_model = LogNormal(np.log(theta[3]), 0.3)
    
    def sample(self, size=None):
        return [self.k_a_model.sample(), self.k_e_model.sample(), self.cl_model.sample(), self.sigma_model.sample()]

    def pdf(self, x):
        return self.k_a_model.pdf(x[0]) * self.k_e_model.pdf(x[1]) * self.cl_model.pdf(x[2]) * self.sigma_model.pdf(x[3])

def compute_expected_theta(coefficients, data):
    """Compute the expected theta from the coefficients and the data (E[theta | D])

    Args:
        coefficients (np.array): Coefficients of the linear regression model (beta_i).
        data (np.array): Data (D).
        
    Returns:
        np.array: The expected theta.
    """
    return np.dot(coefficients, np.hstack([[1], data]).reshape(-1, 1)).squeeze()

class PHKRWDataDrivenProposalModel(Model):
    """Random walk data-driven proposal model for the pharmacokinetics problem."""

    def __init__(self, coefficients, data):
        self.coefficients = coefficients  # estimated coefficients (beta_i) from the linear regression model
        self.data = data
        self.theta = compute_expected_theta(coefficients, data)
        self.k_a_model = LogNormal(np.log(self.theta[0]), 0.4)
        self.k_e_model = LogNormal(np.log(self.theta[1]), 0.6)
        self.cl_model = LogNormal(np.log(self.theta[2]), 0.8)
        self.sigma_model = LogNormal(np.log(self.theta[3]), 0.3)

    def sample(self, size=None):
        return self.theta

    def pdf(self, x):
        return self.k_a_model.pdf(x[0]) * self.k_e_model.pdf(x[1]) * self.cl_model.pdf(x[2]) * self.sigma_model.pdf(x[3])

def make_phk_rw_lognorm_proposal_model(theta, **kwargs):
    return PHKRWLogNormProposalModel(theta)

def make_phk_rw_data_driven_proposal_model(coefficients, theta, data, **kwargs):
    return PHKRWDataDrivenProposalModel(coefficients, data)

def compute_phk_discrepancy(coefficients, theta_0, observed_data, generated_data):
    s_observed_data = compute_expected_theta(coefficients, observed_data)
    s_generated_data = compute_expected_theta(coefficients, generated_data)
    return weighted_euclidean_norm(s_generated_data - s_observed_data, weights=np.array(theta_0))

def generate_phk_data(theta, size):
    phk_model = PHKModel(D=drug_dose, K_a=theta[0], K_e=theta[1], Cl=theta[2], sigma=theta[3], dt=sampling_dt)
    return run_euler_maruyama(sampling_times, phk_model, dt=sampling_dt)

def plot_phk_samples(tolerance_samples, tolerances=phk_tolerances, save_path=None):
    colors = ['blue', 'orange', 'green', 'red', 'black']
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    plt.rcParams['font.size'] = '16'
    params = ['Absorption Rate (K_a)', 'Elimination Rate (K_e)', 'Clearance of the drug (Cl)', 'Intrinsic noise (σ)']
    
    for i in range(4):
        axis = axes[i//2, i%2]
        
        for j, (tol, sample) in enumerate(zip(tolerances, tolerance_samples)):
            sns.kdeplot(sample[:, i], fill=True, color=colors[j], label=f'tol={tol}',ax=axis)
        
        for label in (axis.get_xticklabels() + axis.get_yticklabels()):
            label.set_fontsize(16)

        axis.set_title(f"{params[i]}", fontsize=20)
        axis.set_xlabel("samples")
        axis.set_ylabel("density")
        axis.legend(prop={'size': 14})

    plt.legend()
    plt.show()

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)