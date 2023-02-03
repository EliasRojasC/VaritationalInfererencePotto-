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
rope_length = 1
sd_tie = 0.08
space_h = 10
k_real = 32


# Utils
def cost(h):
    if h > 0:
        return h**2
    else:
        return torch.tensor(100.0)


def generate_samples(count):
    params = (mass, k_real, spring_length, sd_tie, rope_length, space_h)
    sys = sim.srms_pure_python(*params)
    for _ in range(count):
        s, l, p = sys.bounce_height_stochastic(0.001)
        yield s


# def normalpdf(mu, sigma):
#     return lambda x: (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp((-1 / 2) * (((x - mu) / (sigma)) ** 2))


# Future Implements


def model(k):
    params = (mass, k, spring_length, sd_tie, rope_length, space_h)
    s, l, p = sim.srms_pure_python(*params)
    return cost(s.bounce_height_stochastic(time_step))


# Var Inf Setup


def prior():
    return lambda c: cost((1 / (100 * torch.sqrt(torch.tensor(2 * math.pi)))) * torch.exp(
        (-1 / 2) * (((c - 50) / (100)) ** 2)
    ))


def joint():
    def breakout(z, x):
        j = list(map(prior(), z))

        def diff(c, z):
            constant = 1 / (10 * torch.sqrt(torch.tensor(2 * math.pi)))
            temp = z[0]
            in_f = (-1 / 2) * (((c - temp) / (10)) ** 2)
            non_zero_f = torch.exp(in_f)
            return constant * non_zero_f + torch.tensor(10 ** (-8))

        k = []
        for i in x:
            t = diff(i, z)
            k.append(t)
        last = j[0] * torch.prod(torch.tensor(k)) + torch.tensor(10 ** (-8))
        return last

    return breakout

#http://proceedings.mlr.press/v89/xu19a/xu19a.pdf
#https://arxiv.org/pdf/1312.6114.pdf
def guide():
    return lambda k, mu, sigma: 5.0 + mu + k * sigma


def ELBO_loss(j, g, m_p, obs):
    mu, sigma = m_p
    num_samples = 1000
    s_points = np.random.normal(0, 1, num_samples)

    def elbo_exp(c, m, s):
        j_c = j()([m + c[0] * s], obs)
        q_c = g()(c[0], mu, sigma)
        return torch.log(j_c) - torch.log(cost(q_c))

    expectation_exp = lambda x: elbo_exp((x,), mu, sigma)
    t = torch.tensor(0.0)
    for i in s_points:
        t += expectation_exp(i) #WE WANT TO MAXIMIZE THE ELBO SO THIS SHOULD BE ADATIVE
    int_result = t / len(s_points)
    return int_result

"""
https://chrisorm.github.io/VI-MC.html
"""


def learn(samples, itrs, gamma):
    mu = torch.tensor(100.0, requires_grad=True)
    sigma = torch.tensor(100.0, requires_grad=True)
    for _ in trange(itrs):
        l = ELBO_loss(joint, guide, (mu, sigma), samples)
        l.backward()
        dmu = mu.grad
        dsig = sigma.grad
        print(f"Loss: {l}, dmu: {dmu}, dsigma: {dsig}, mu: {mu}, sigma {sigma}")
        with torch.no_grad():
            mu += gamma * dmu
        mu.grad.zero_()
        sigma.grad.zero_()


if __name__ == "__main__":
    s = list(generate_samples(10))
    learn(s, 50, 100)
