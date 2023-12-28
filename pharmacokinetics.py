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

def compute_cov(history, c_0, t=0, s_d=1, eps=0.001, cache=None):
    if t <= 1:
        return np.array(c_0)
    
    if cache and t in cache:
        return cache[t]

    x_bar_t_minus_1 = np.mean(history[:t-1], axis=0).reshape(-1, 1)
    x_bar_t = np.mean(history[:t], axis=0).reshape(-1, 1)
    x_t = history[t-1].reshape(-1, 1)
    c_t = ((t-1)/t) * compute_cov(history, c_0, t-1, s_d, eps) + (s_d/t) * (t * np.dot(x_bar_t_minus_1, x_bar_t_minus_1.T) - (t+1) * np.dot(x_bar_t, x_bar_t.T) + np.dot(x_t, x_t.T)) + eps * np.eye(len(x_t))
    
    if cache:
        cache[t] = c_t

    return c_t

class PharmacokineticProposalModel(Model):
    def __init__(self, theta_history, s_d=2.4*2.4, eps=0.001):
        self.s_d = s_d
        self.eps = eps
        self.mean = theta_history[-1]
        self.k_a_mean = self.mean[0]
        self.k_e_mean = self.mean[1]
        self.cl_mean = self.mean[2]
        self.sigma_mean = self.mean[3]
        self.k_a_cov = compute_cov(theta_history[:, 0], c_0=0.4, t=len(theta_history), s_d=self.s_d, eps=self.eps).item()
        self.k_e_cov = compute_cov(theta_history[:, 1], c_0=0.6, t=len(theta_history), s_d=self.s_d, eps=self.eps).item()
        self.cl_cov = compute_cov(theta_history[:, 2], c_0=0.8, t=len(theta_history), s_d=self.s_d, eps=self.eps).item()
        self.sigma_cov = compute_cov(theta_history[:, 3], c_0=0.3, t=len(theta_history), s_d=self.s_d, eps=self.eps).item()
        self.d = 1

    def sample(self, size=None):
        return [stats.norm.rvs(self.k_a_mean, self.k_a_cov, size=size), 
                stats.norm.rvs(self.k_e_mean, self.k_e_cov, size=size), 
                stats.norm.rvs(self.cl_mean, self.cl_cov, size=size), 
                stats.norm.rvs(self.sigma_mean, self.sigma_cov, size=size)]

    def pdf(self, x):
        return stats.norm.pdf(x[0], self.k_a_mean, self.k_a_cov) * stats.norm.pdf(x[1], self.k_e_mean, self.k_e_cov) * stats.norm.pdf(x[2], self.cl_mean, self.cl_cov) * stats.norm.pdf(x[3], self.sigma_mean, self.sigma_cov)

def make_pharmacokinetic_proposal_model(theta, theta_history, window_size=100):
    if len(theta_history) == 0:
        return PharmacokineticProposalModel(np.array([theta]))
    return PharmacokineticProposalModel(np.vstack([theta_history[-window_size:], theta]))
    
def compute_pharmacokinetics_discrepancy(coefficients, theta_0, observed_data, generated_data):
    s_observed_data = np.dot(coefficients, np.hstack([[1], observed_data]).reshape(-1, 1))
    s_generated_data = np.dot(coefficients, np.hstack([[1], generated_data]).reshape(-1, 1))
    return weighted_euclidean_norm(s_generated_data - s_observed_data, weights=theta_0)

def generate_pharmacokinetics_data(theta, size):
    pharmacokinetic_model = PharmacokineticModel(D=4, K_a=theta[0], K_e=theta[1], Cl=theta[2], sigma=theta[3], dt=sampling_dt)
    return run_euler_maruyama(sampling_times, pharmacokinetic_model, dt=sampling_dt)