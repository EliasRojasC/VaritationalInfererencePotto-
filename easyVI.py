import math

from tqdm import trange

import sim
from numpy.random import normal
import numpy as np
import torch
import torch.distributions.constraints as constraints

mass = 1
time_step = 0.1
spring_length = 1
rope_length = 9.7
sd_tie = 0.08
space_h = 10
k_real = 32


# Utils
def cost(h):
    if h > 0:
        return h**2
    else:
        return torch.tensor(100.0)

def real_landing(h): return h if h > 0 else 0

def loss(mu, sigma, rope_length, space_height, x):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * torch.exp((-1 / 2) * ((((space_height - rope_length - x) - mu) / (sigma)) ** 2))

def learn(itrs, gamma_mu, gamma_sigma):
    mu = torch.tensor(1.0, requires_grad=True)
    sigma = torch.tensor(1.0, requires_grad=True)
    for _ in trange(itrs):
        params = (mu, sigma, rope_length, space_h, 0.1)
        l = loss(*params)
        l.backward()
        dmu = mu.grad
        with torch.no_grad():
            mu += gamma_mu * dmu
        mu.grad.zero_()
        sigma.grad.zero_()

    for _ in trange(itrs):
        params = (mu, sigma, rope_length, space_h, 0.1)
        l = loss(*params)
        l.backward()
        dsig = sigma.grad
        with torch.no_grad():
            if sigma + gamma_sigma * dsig > 0:
                sigma += gamma_sigma * dsig
            else:
                gamma_sigma = gamma_sigma/2
        mu.grad.zero_()
        sigma.grad.zero_()
    return (mu, sigma)


if __name__ == "__main__":
    print(learn(10000, 0.1, 0.001))
