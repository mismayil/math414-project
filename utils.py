from abc import ABC, abstractmethod
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_samples(samples, posterior_model, tolerances, set_log=True, set_ylim=None):
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    for i, sample in enumerate(samples):
        axis = axes[i//2, i%2]
        axis.hist(sample, bins=100, label=f"tolerance={tolerances[i]}", density=True)
        x = np.linspace(min(sample), max(sample), len(sample))
        posterior_density = [posterior_model.pdf(theta) for theta in x]
        axis.plot(x, posterior_density, label="true posterior")
        if set_log:
            axis.set_yscale("log")
        if set_ylim is not None:
            axis.set_ylim(bottom=set_ylim)
        axis.legend()
    plt.show()

class Model(ABC):
    @abstractmethod
    def sample(self, size=None, *args, **kwargs):
        pass

    @abstractmethod
    def pdf(self, x):
        pass

class Normal(Model):
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
    
    def sample(self, size=None):
        return stats.norm.rvs(self.mu, self.sigma, size=size)

    def pdf(self, x):
        return stats.norm.pdf(x, self.mu, self.sigma)

class LogNormal(Model):
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
    
    def sample(self, size=None):
        return np.exp(stats.norm.rvs(loc=self.mu, scale=self.sigma, size=size))

    def pdf(self, x):
        return np.exp(stats.norm.pdf(x, self.mu, self.sigma))

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

def run_abc_mcmc(N, observed_data, make_proposal_model, prior_model, generate_data, compute_discrepancy, tolerance=0.1, theta_0=0, burn_in=0.1):
    sample = [theta_0]
    num_accepted = 0
    burn_in_size = int(N * burn_in)

    for i in tqdm(range(burn_in_size+N), desc="Generating samples"):
        current_theta = sample[-1]
        current_proposal_model = make_proposal_model(current_theta)
        new_theta = current_proposal_model.sample()
        new_proposal_model = make_proposal_model(new_theta)
        generated_data = generate_data(new_theta, len(observed_data))  
        
        if compute_discrepancy(observed_data, generated_data) < tolerance:
            alpha = min(1, (prior_model.pdf(new_theta) * new_proposal_model.pdf(current_theta)) / (prior_model.pdf(current_theta) * current_proposal_model.pdf(new_theta)))
            prob = stats.uniform.rvs(0, 1)
            if prob < alpha:
                sample.append(new_theta)
                num_accepted += 1
            else:
                sample.append(current_theta)
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