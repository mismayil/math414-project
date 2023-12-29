import scipy.stats as stats
import numpy as np

from utils import Model, Normal

example_a = 1
example_p = 1/2
example_M = 100
example_sigma_1 = np.sqrt(0.1)
example_sigma = np.sqrt(3)
example_mean = 0
example_N = 500
example_tolerances = [0.75, 0.25, 0.1, 0.025]

class ExampleLikelihoodModel(Model):
    def __init__(self, theta, a, sigma, p):
        self.theta = theta
        self.a = a
        self.sigma = sigma
        self.p = p
        self.dist1 = stats.norm(loc=self.theta, scale=self.sigma)
        self.dist2 = stats.norm(loc=self.theta+self.a, scale=self.sigma)
    
    def sample(self, size=None):
        prob = stats.uniform.rvs(0, 1)

        if prob < self.p:
            return self.dist1.rvs(size=size)

        return self.dist2.rvs(size=size)

    def pdf(self, x):
        return NotImplementedError

class ExamplePosteriorModel(Model):
    def __init__(self, M, mean, a, sigma, sigma_1):
        self.a = a
        self.sigma = sigma
        self.sigma_1 = sigma_1
        self.mean = mean
        self.M = M
        self.alpha = 1 / (1 + np.exp(a * (mean - a/2) * (self.M / (self.M*sigma**2 + sigma_1**2))))
        self.normal_1 = Normal((sigma**2 / (sigma**2 + sigma_1**2/self.M)) * mean, sigma_1**2/(self.M + sigma_1**2/sigma**2))
        self.normal_2 = Normal((sigma**2 / (sigma**2 + sigma_1**2/self.M)) * (mean-a), sigma_1**2/(self.M + sigma_1**2/sigma**2))
    
    def sample(self, size=None):
        return NotImplementedError
    
    def pdf(self, x):
        return self.alpha * self.normal_1.pdf(x) + (1-self.alpha) * self.normal_2.pdf(x)

class ExampleProposalModel(Model):
    def __init__(self, theta, sigma):
        self.theta = theta
        self.sigma = sigma
        self.dist = stats.norm(loc=self.theta, scale=self.sigma)
    
    def sample(self, size=None):
        return self.dist.rvs(size=size)

    def pdf(self, x):
        return self.dist.pdf(x)

def make_example_proposal_model(theta, sigma=np.sqrt(0.1), *args, **kwargs):
    return ExampleProposalModel(theta, sigma)

def generate_example_data(theta, size):
    return ExampleLikelihoodModel(theta, example_a, example_sigma_1, example_p).sample(size=size)

def compute_example_discrepancy(observed_data, generated_data):
    return np.abs(np.mean(generated_data) - example_mean)