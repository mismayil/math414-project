from utils import Model
import scipy.stats as stats
import numpy as np

from utils import Normal, LogNormal, MultivariateNormal, weighted_euclidean_norm, run_euler_maruyama

sampling_dt = 0.005
sampling_times = [0.25, 0.5, 1, 2, 3.5, 5, 7, 9, 12]
train_size = 1000
drug_dose = 4

class BrownianMotion(Model):
    def __init__(self, dt=0.01) -> None:
        self.dt = dt
        self.dist = stats.norm(0, np.sqrt(self.dt))

    def sample(self, size=None):
        return self.dist.rvs(size=size)
    
    def pdf(self, x):
        return self.dist.pdf(x)

class PHKModel(Model):
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

class PHKPriorModel(Model):
    def __init__(self):
        self.prior_K_a = LogNormal(0.14, 0.4)
        self.prior_K_e = LogNormal(-2.7, 0.6)
        self.prior_Cl = LogNormal(-3, 0.8)
        self.prior_sigma = LogNormal(-1.1, 0.3)
    
    def sample(self, size=None):
        return [self.prior_K_a.sample(), self.prior_K_e.sample(), self.prior_Cl.sample(), self.prior_sigma.sample()]

    def pdf(self, x):
        return self.prior_K_a.pdf(x[0]) * self.prior_K_e.pdf(x[1]) * self.prior_Cl.pdf(x[2]) * self.prior_sigma.pdf(x[3])

def compute_cov(history, c_0, t=0, t_0=1, s_d=1, eps=0.001, cache=None):
    if t <= t_0:
        return np.array(c_0)
    
    if cache and t in cache:
        return cache[t]

    x_bar_t_minus_1 = np.mean(history[:t-1], axis=0).reshape(-1, 1)
    x_bar_t = np.mean(history[:t], axis=0).reshape(-1, 1)
    x_t = history[t-1].reshape(-1, 1)
    c_t = ((t-2)/(t-1)) * compute_cov(history, c_0, t-1, t_0, s_d, eps) + (s_d/(t-1)) * ((t-1) * np.dot(x_bar_t_minus_1, x_bar_t_minus_1.T) - t * np.dot(x_bar_t, x_bar_t.T) + np.dot(x_t, x_t.T)) + eps * np.eye(len(x_t))
    
    if cache:
        cache[t] = c_t

    return c_t

class PHKRWLogNormProposalModel(Model):
    def __init__(self, theta):
        self.k_a_model = LogNormal(np.log(theta[0]), 0.4)
        self.k_e_model = LogNormal(np.log(theta[1]), 0.6)
        self.cl_model = LogNormal(np.log(theta[2]), 0.8)
        self.sigma_model = LogNormal(np.log(theta[3]), 0.3)
    
    def sample(self, size=None):
        return [self.k_a_model.sample(), self.k_e_model.sample(), self.cl_model.sample(), self.sigma_model.sample()]

    def pdf(self, x):
        return self.k_a_model.pdf(x[0]) * self.k_e_model.pdf(x[1]) * self.cl_model.pdf(x[2]) * self.sigma_model.pdf(x[3])

class PHKRWNormProposalModel(Model):
    def __init__(self, theta):
        self.k_a_model = Normal(np.log(theta[0]), 0.4)
        self.k_e_model = Normal(np.log(theta[1]), 0.6)
        self.cl_model = Normal(np.log(theta[2]), 0.8)
        self.sigma_model = Normal(np.log(theta[3]), 0.3)
    
    def sample(self, size=None):
        return [np.exp(self.k_a_model.sample()), np.exp(self.k_e_model.sample()), np.exp(self.cl_model.sample()), np.exp(self.sigma_model.sample())]

    def pdf(self, x):
        return self.k_a_model.pdf(x[0]) * self.k_e_model.pdf(x[1]) * self.cl_model.pdf(x[2]) * self.sigma_model.pdf(x[3])

class PHKRWMultiNormProposalModel(Model):
    def __init__(self, theta):
        self.model = MultivariateNormal(np.log(theta), np.diag(np.square([0.4, 0.6, 0.8, 0.3])))
    
    def sample(self, size=None):
        return np.exp(self.model.sample(size=size))

    def pdf(self, x):
        return self.model.pdf(x)

def compute_expected_theta(coefficients, data):
    return np.dot(coefficients, np.hstack([[1], data]).reshape(-1, 1)).squeeze()

class PHKRWDataDrivenProposalModel(Model):
    def __init__(self, coefficients, data):
        self.coefficients = coefficients
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

class PHKAdaptiveProposalModel(Model):
    def __init__(self, theta_history, t_0=1, s_d=2.4*2.4, eps=0.001):
        self.s_d = s_d
        self.eps = eps
        self.t_0 = t_0
        self.mean = theta_history[-1]
        self.k_a_mean = self.mean[0]
        self.k_e_mean = self.mean[1]
        self.cl_mean = self.mean[2]
        self.sigma_mean = self.mean[3]
        self.k_a_cov = compute_cov(theta_history[:, 0], c_0=np.square(0.4), t=len(theta_history), t_0=t_0, s_d=self.s_d, eps=self.eps).item()
        self.k_e_cov = compute_cov(theta_history[:, 1], c_0=np.square(0.6), t=len(theta_history), t_0=t_0, s_d=self.s_d, eps=self.eps).item()
        self.cl_cov = compute_cov(theta_history[:, 2], c_0=np.square(0.8), t=len(theta_history), t_0=t_0, s_d=self.s_d, eps=self.eps).item()
        self.sigma_cov = compute_cov(theta_history[:, 3], c_0=np.square(0.3), t=len(theta_history), t_0=t_0, s_d=self.s_d, eps=self.eps).item()
        self.k_a_model = LogNormal(np.log(self.k_a_mean), np.sqrt(self.k_a_cov))
        self.k_e_model = LogNormal(np.log(self.k_e_mean), np.sqrt(self.k_e_cov))
        self.cl_model = LogNormal(np.log(self.cl_mean), np.sqrt(self.cl_cov))
        self.sigma_model = LogNormal(np.log(self.sigma_mean), np.sqrt(self.sigma_cov))
        self.d = 1

    def sample(self, size=None):
        return [self.k_a_model.sample(), self.k_e_model.sample(), self.cl_model.sample(), self.sigma_model.sample()]

    def pdf(self, x):
        return self.k_a_model.pdf(x[0]) * self.k_e_model.pdf(x[1]) * self.cl_model.pdf(x[2]) * self.sigma_model.pdf(x[3])

def make_phk_rw_lognorm_proposal_model(theta, **kwargs):
    return PHKRWLogNormProposalModel(theta)

def make_phk_rw_norm_proposal_model(theta, **kwargs):
    return PHKRWNormProposalModel(theta)

def make_phk_rw_multi_norm_proposal_model(theta, **kwargs):
    return PHKRWMultiNormProposalModel(theta)

def make_phk_rw_data_driven_proposal_model(coefficients, theta, data, **kwargs):
    return PHKRWDataDrivenProposalModel(coefficients, data)

def make_phk_adaptive_proposal_model(theta, theta_history, t_0=1, window_size=100, **kwargs):
    if len(theta_history) == 0:
        return PHKAdaptiveProposalModel(theta_history=np.array([theta]), t_0=t_0)
    return PHKAdaptiveProposalModel(theta_history=np.vstack([theta_history[-window_size:], theta]), t_0=t_0)

def compute_phk_discrepancy(coefficients, theta_0, observed_data, generated_data):
    s_observed_data = compute_expected_theta(coefficients, observed_data)
    s_generated_data = compute_expected_theta(coefficients, generated_data)
    return weighted_euclidean_norm(s_generated_data - s_observed_data, weights=np.array(theta_0))

def generate_phk_data(theta, size):
    phk_model = PHKModel(D=drug_dose, K_a=theta[0], K_e=theta[1], Cl=theta[2], sigma=theta[3], dt=sampling_dt)
    return run_euler_maruyama(sampling_times, phk_model, dt=sampling_dt)