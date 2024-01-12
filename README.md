# Approximate Bayesian Computation (ABC) Project
### Mahammad Ismayilzada, MATH-414, Stochastic Simulation

This repo contains the code and data for the ABC project report. 

## Setup
To setup the environment, please use `python>=3.10` and install requirements using the `requirements.txt` file:
```sh
pip install -r requirements.txt
```

## Repo structure
- `utils.py`: This file contains code for common utilities.
- `example.py`: This file contains common code for the synthetic (example) problem.
- `pharmacokinetics.py`: This file contains common code for the pharmacokinetics problem.
- `q{n}.py`: This file contains the code to run experiments for question `n` in the project description.
- `data`: This folder contains the experiment results.
- `project.ipynb`: This interactive notebook contains code for inspection and visualization of experiment results.