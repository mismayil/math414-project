from abc import ABC
import scipy.stats as stats
import numpy as np

from tqdm import tqdm

class Model(ABC):
    def sample(self, size=None, *args, **kwargs):
        raise NotImplementedError

    def pdf(self, x):
        raise NotImplementedError

class Normal(Model):
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        self.dist = stats.norm(self.mu, self.sigma)
    
    def sample(self, size=None):
        return self.dist.rvs(size=size)

    def pdf(self, x):
        return self.dist.pdf(x)

class LogNormal(Model):
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        self.dist = stats.lognorm(s=self.sigma, scale=np.exp(self.mu))
    
    def sample(self, size=None):
        return self.dist.rvs(size=size)

    def pdf(self, x):
        return self.dist.pdf(x)

class MultivariateNormal(Model):
    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov
        self.dist = stats.multivariate_normal(self.mu, self.cov)
    
    def sample(self, size=None):
        return self.dist.rvs(size=size)

    def pdf(self, x):
        return self.dist.pdf(x)
    
def run_abc_rejection(N, observed_data, prior_model, generate_data, compute_discrepancy, tolerance=0.1):
    sample = []
    num_tries = 0

    with tqdm(total=N, desc="Generating samples") as pbar:
        while len(sample) < N:
            num_tries += 1
            theta = prior_model.sample()
            generated_data = generate_data(theta, len(observed_data))
            if compute_discrepancy(observed_data, generated_data) < tolerance:
                sample.append(theta)
                pbar.update(1)
    
    return sample, N / num_tries

def run_abc_mcmc(N, observed_data, make_proposal_model, prior_model, generate_data, compute_discrepancy, tolerance=0.1, theta_0=0, burn_in=0.1, data_0=0, debug=False):
    sample = [theta_0]
    sample_data = [data_0]
    num_accepted = 0
    burn_in_size = int(N * burn_in)

    for i in tqdm(range(burn_in_size+N), desc="Generating samples"):
        current_theta = sample[-1]
        current_proposal_model = make_proposal_model(theta=current_theta, data=sample_data[-1])
        new_theta = current_proposal_model.sample()
        generated_data = generate_data(new_theta, len(observed_data))
        new_proposal_model = make_proposal_model(theta=new_theta, data=generated_data) 

        if compute_discrepancy(observed_data, generated_data) < tolerance:
            alpha = min(1, (prior_model.pdf(new_theta) * new_proposal_model.pdf(current_theta)) / (prior_model.pdf(current_theta) * current_proposal_model.pdf(new_theta)))
            prob = stats.uniform.rvs()
            if prob < alpha:
                sample.append(new_theta)
                sample_data.append(generated_data)
                num_accepted += 1
            else:
                sample.append(current_theta)
            if debug:
                print(f"i={i}, theta={new_theta}, alpha={alpha}, prob={prob}, accepted={prob < alpha}")
        else:
            sample.append(current_theta)
    
    return sample[burn_in_size+1:], num_accepted / (N+burn_in_size)

def run_euler_maruyama(sampling_times, model, x_0=0, dt=0.01, debug=False):
    x = [x_0]
    sample = []
    t = 0

    while len(sample) < len(sampling_times):
        x_t = x[-1]
        x_t_plus_1 = x_t + model.sample(x_t=x_t, t=t)
        x.append(x_t_plus_1)

        if any([np.isclose(t, sampling_time) for sampling_time in sampling_times]):
            if debug:
                print(f"t={t}, x_t={x_t_plus_1}")
            sample.append(x_t_plus_1)
        
        t = t + dt

    return sample

def weighted_euclidean_norm(x, weights=1):
    return np.sqrt(np.sum(np.square(x)/np.square(weights)))