import math

from tqdm import trange

import sim
from numpy.random import normal
import numpy as np
import torch
import torch.distributions.constraints as constraints
import matplotlib.pyplot as plt

mass = 1
time_step = 0.1
spring_length = 5
rope_length = 0
sd_tie = 0.08
space_h = 10
k_real = 32

def loss(mu, sigma, rope_length, space_height, x):
    sys = sim.srms_torch(mass, k_real, spring_length, 0, rope_length, space_height)
    return sys.bounce_height_deterministic(0.1, mu, sigma)


def learn(itrs, gamma_mu, gamma_sigma):
    mu = torch.tensor(1.0, requires_grad=True)
    sigma = torch.tensor(1.0, requires_grad=True)
    mu_s = []
    sigma_s = []
    for _ in trange(itrs):
        params = (mu, sigma, rope_length, space_h, 0.1)
        l = loss(*params)
        l.backward()
        dmu = mu.grad
        with torch.no_grad():
            mu += gamma_mu * dmu
        mu.grad.zero_()
        sigma.grad.zero_()
        mu_s.append(mu.detach().numpy().copy())

    for _ in trange(itrs):
        params = (mu, sigma, rope_length, space_h, 0.1)
        l = loss(*params)
        l.backward()
        dsig = sigma.grad
        with torch.no_grad():
            if sigma + gamma_sigma * dsig > 0:
                sigma += gamma_sigma * dsig
            else:
                gamma_sigma = gamma_sigma / 2
        mu.grad.zero_()
        sigma.grad.zero_()
        sigma_s.append(sigma.detach().numpy().copy())
    return (mu, sigma, mu_s)


if __name__ == "__main__":
    m, s, g = learn(200, 0.1, 0.1)
    a = [x for x in range(len(g))]
    plt.plot(a, g, 'o', color='black')
    plt.xscale("log")
    plt.show()
    print(g)
