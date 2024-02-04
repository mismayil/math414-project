"""
This file contains common utility code such as implementations of statistical models and ABC algorithms.
"""
from abc import ABC
import scipy.stats as stats
import numpy as np
from typing import Union, Callable, List, Tuple, Any
import statsmodels.api as sm

from tqdm import tqdm

class Model(ABC):
    """Abstract class for statistical models."""

    def sample(self, size=None, *args, **kwargs):
        """Sample from the model.

        Args:
            size (int, optional): Number of samples to generate. Defaults to None.

        Returns:
            float: The sample.
        """
        raise NotImplementedError

    def pdf(self, x):
        """Compute the probability density function of the model.

        Args:
            x (float): Value to compute the pdf at.
        
        Returns:
            float: The pdf value.
        """
        raise NotImplementedError

class Normal(Model):
    """Normal distribution."""

    def __init__(self, mu=0, sigma=1):
        """Initialize the model.
        
        Args:
            mu (float, optional): Mean of the distribution. Defaults to 0.
            sigma (float, optional): Standard deviation of the distribution. Defaults to 1.
        """
        self.mu = mu
        self.sigma = sigma
        self.dist = stats.norm(loc=self.mu, scale=self.sigma)
    
    def sample(self, size=None):
        return self.dist.rvs(size=size)

    def pdf(self, x):
        return self.dist.pdf(x)

class LogNormal(Model):
    """Log-normal distribution."""

    def __init__(self, mu=0, sigma=1):
        """Initialize the model.

        Args:
            mu (float, optional): Mean of the underlying normal distribution. Defaults to 0.
            sigma (float, optional): Standard deviation of the underlying normal distribution. Defaults to 1.
        """
        self.mu = mu
        self.sigma = sigma
        self.dist = stats.lognorm(s=self.sigma, scale=np.exp(self.mu))
    
    def sample(self, size=None):
        return self.dist.rvs(size=size)

    def pdf(self, x):
        return self.dist.pdf(x)
    
def run_abc_rejection(N: int, observed_data: Union[List, np.array], prior_model: Model,
                      generate_data: Callable, compute_discrepancy: Callable, tolerance: float = 0.1) -> Tuple[List, float]:
    """Run the ABC rejection algorithm.

    Args:
        N (int): Number of samples to generate.
        observed_data (Union[List, np.array]): Observed data.
        prior_model (Model): Prior model of the samples.
        generate_data (Callable): Function to generate data from given parameters.
        compute_discrepancy (Callable): Function to compute the discrepancy between two sets of data.
        tolerance (float, optional): Tolerance for the discrepancy. Defaults to 0.1.
    
    Returns:
        Tuple[List, float]: The generated samples and acceptance rate.
    """
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
    
    acceptance_rate = N / num_tries
    
    return sample, acceptance_rate

def compute_ess(samples: List, ess_lag: int = 1) -> List[int]:
    """Compute the effective sample size.

    Args:
        samples (List): List of samples.
        ess_lag (int, optional): Number of lags to use for the effective sample size. Defaults to 1.
    
    Returns:
        List[int]: The effective sample sizes for each dimension of samples.
    """
    samples = np.array(samples)
    ess_sizes = []

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    if ess_lag == 0:
        return np.array([len(samples)] * samples.shape[1])

    for dim in range(samples.shape[1]):
        acorr = sm.tsa.acf(samples[:, dim], fft=True, nlags=ess_lag)
        ess = len(samples) / (1 + 2 * np.sum(acorr[1:]))
        ess_sizes.append(int(ess))

    return np.array(ess_sizes)

def run_abc_mcmc(N: int, observed_data: Union[List, np.array], make_proposal_model: Callable, 
                 prior_model: Model, generate_data: Callable, compute_discrepancy: Callable, 
                 tolerance: float = 0.1, theta_0: Any = 0, burn_in: float = 0.1, ess_lag: int = 1, data_0: Any = 0,
                 debug: bool = False) -> Tuple[List, float]:
    """Run the ABC MCMC algorithm.

    Args:
        N (int): Number of samples to generate.
        observed_data (Union[List, np.array]): Observed data.
        make_proposal_model (Callable): Function to make the proposal model given the current parameters.
        prior_model (Model): Prior model of the samples.
        generate_data (Callable): Function to generate data from given parameters.
        compute_discrepancy (Callable): Function to compute the discrepancy between two sets of data.
        tolerance (float, optional): Tolerance for the discrepancy. Defaults to 0.1.
        theta_0 (Any, optional): Initial parameter value. Defaults to 0.
        burn_in (float, optional): Burn-in ratio. Defaults to 0.1.
        ess_lag (int, optional): Number of lags to use for the effective sample size. Defaults to 1.
        data_0 (Any, optional): Initial data value. Defaults to 0.
        debug (bool, optional): Whether to print debug information. Defaults to False.
    
    Returns:
        Tuple[List, float]: The generated samples and acceptance rate.
    """
    sample = [theta_0]
    sample_data = [data_0]
    num_accepted = 0
    num_tries = 0
    ess = 0
    burn_in_size = int(N * burn_in)
    total_N = N + burn_in_size

    with tqdm(total=total_N, desc="Generating samples") as pbar:
        while True:
            num_tries += 1
            current_theta = sample[-1]
            current_data = sample_data[-1]
            
            # Define q(theta, ·)
            current_proposal_model = make_proposal_model(theta=current_theta, data=current_data)

            # Sample theta* from q(theta, ·)
            new_theta = current_proposal_model.sample()

            # Generate data from theta*
            generated_data = generate_data(new_theta, len(observed_data))

            # Define q(theta*, ·)
            new_proposal_model = make_proposal_model(theta=new_theta, data=generated_data) 

            if compute_discrepancy(observed_data, generated_data) < tolerance:
                # Compute acceptance probability
                alpha = min(1, (prior_model.pdf(new_theta) * new_proposal_model.pdf(current_theta)) / (prior_model.pdf(current_theta) * current_proposal_model.pdf(new_theta)))
                prob = stats.uniform.rvs()
                if prob < alpha:
                    sample.append(new_theta)
                    sample_data.append(generated_data)
                    num_accepted += 1
                else:
                    sample.append(current_theta)
                if debug:
                    print(f"theta={new_theta}, alpha={alpha}, prob={prob}, accepted={prob < alpha}")
            else:
                sample.append(current_theta)

            # Compute effective sample size once we have enough samples
            if len(sample) >= total_N:
                new_ess = compute_ess(sample[burn_in_size:], ess_lag)
                
                if num_tries == total_N:
                    pbar.clear()

                pbar.update(max(0, np.min(new_ess - ess)))
                ess = new_ess

            # Break if we have enough effective samples
            if np.all(ess >= N):
                break

            if num_tries < total_N:
                pbar.update(1)

    acceptance_rate = num_accepted / num_tries

    return sample[burn_in_size+1:], acceptance_rate

def run_euler_maruyama(sampling_times: List[int], model: Model, x_0: Any = 0, dt: float = 0.01, debug: bool = False) -> List:
    """Run the Euler-Maruyama algorithm.

    Args:
        sampling_times (List[int]): List of sampling times.
        model (Model): Model to sample from.
        x_0 (Any, optional): Initial value. Defaults to 0.
        dt (float, optional): Time step. Defaults to 0.01.
        debug (bool, optional): Whether to print debug information. Defaults to False.
    
    Returns:
        List: The generated samples.
    """
    x = [x_0]
    sample = []
    t = 0

    while len(sample) < len(sampling_times):
        x_t = x[-1]
        x_t_plus_1 = x_t + model.sample(x_t=x_t, t=t)
        x.append(x_t_plus_1)

        # Check if the current timestep is a sampling time that we want to record at
        if any([np.isclose(t, sampling_time) for sampling_time in sampling_times]):
            if debug:
                print(f"t={t}, x_t={x_t_plus_1}")
            sample.append(x_t_plus_1)
        
        t = t + dt

    return sample

def weighted_euclidean_norm(x: np.array, weights: np.array = 1) -> float:
    """Compute the weighted Euclidean norm.

    Args:
        x (np.array): Array of values.
        weights (np.array, optional): Array of weights. Defaults to 1.
    
    Returns:
        float: The weighted Euclidean norm.
    """
    return np.sqrt(np.sum(np.square(x)/np.square(weights)))