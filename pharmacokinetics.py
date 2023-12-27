from utils import Model
import scipy.stats as stats
import numpy as np

from utils import LogNormal,weighted_euclidean_norm, run_euler_maruyama

sampling_dt = 0.005
sampling_times = [0.25, 0.5, 1, 2, 3.5, 5, 7, 9, 12]
train_size = 500
drug_dose = 4

class BrownianMotion(Model):
    def __init__(self, dt=0.01) -> None:
        self.dt = dt

    def sample(self, size=None):
        return stats.norm.rvs(0, np.sqrt(self.dt), size=size)
    
    def pdf(self, x):
        return NotImplementedError

class PharmacokineticModel(Model):
    def __init__(self, D, K_a, K_e, Cl, sigma=1, dt=0.01):
        self.D = D
        self.K_a = K_a
        self.K_e = K_e
        self.Cl = Cl
        self.sigma = sigma
        self.dt = dt
        self.brownian = BrownianMotion(dt=dt)
    
    def sample(self, size=None, x_t=0, t=0.01):
        return ((self.D * self.K_a * self.K_e * np.exp(-self.K_a * t))/self.Cl - self.K_e * x_t) * self.dt + self.sigma * self.brownian.sample(size=size)

    def pdf(self, x):
        return NotImplementedError

class PharmacokineticPriorModel(Model):
    def __init__(self):
        self.prior_K_a = LogNormal(0.14, 0.4)
        self.prior_K_e = LogNormal(-2.7, 0.6)
        self.prior_Cl = LogNormal(-3, 0.8)
        self.prior_sigma = LogNormal(-1.1, 0.3)
    
    def sample(self, size=None):
        return [self.prior_K_a.sample(), self.prior_K_e.sample(), self.prior_Cl.sample(), self.prior_sigma.sample()]

    def pdf(self, x):
        return self.prior_K_a.pdf(x[0]) * self.prior_K_e.pdf(x[1]) * self.prior_Cl.pdf(x[2]) * self.prior_sigma.pdf(x[3])

class PharmacokineticProposalModel(Model):
    def __init__(self, theta):
        self.k_a_model = LogNormal(theta[0], 0.4)
        self.k_e_model = LogNormal(theta[1], 0.6)
        self.cl_model = LogNormal(theta[2], 0.8)
        self.sigma_model = LogNormal(theta[3], 0.3)
    
    def sample(self, size=None):
        return [self.k_a_model.sample(), self.k_e_model.sample(), self.cl_model.sample(), self.sigma_model.sample()]

    def pdf(self, x):
        return self.k_a_model.pdf(x[0]) * self.k_e_model.pdf(x[1]) * self.cl_model.pdf(x[2]) * self.sigma_model.pdf(x[3])

def make_pharmacokinetic_proposal_model(theta):
    return PharmacokineticProposalModel(theta)
    
def compute_pharmacokinetics_discrepancy(coefficients, theta_0, observed_data, generated_data):
    s_observed_data = np.dot(coefficients, np.hstack([[1], observed_data]).reshape(-1, 1))
    s_generated_data = np.dot(coefficients, np.hstack([[1], generated_data]).reshape(-1, 1))
    return weighted_euclidean_norm(s_generated_data - s_observed_data, weights=theta_0)

def generate_pharmacokinetics_data(theta, size):
    pharmacokinetic_model = PharmacokineticModel(D=4, K_a=theta[0], K_e=theta[1], Cl=theta[2], sigma=theta[3], dt=sampling_dt)
    return run_euler_maruyama(sampling_times, pharmacokinetic_model, dt=sampling_dt)