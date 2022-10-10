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


#Utils
def cost(h):
    if h > 0:
        return h**2
    else:
        return 100

def generate_samples(count):
    params = (mass, k_real, spring_length, sd_tie, rope_length, space_h)
    sys = sim.spring_rope_mass_system(*params)
    for _ in range(count):
        s,l,p = sys.bounce_height_stochastic(0.001)
        yield s

def normalpdf(mu, sigma):
    return lambda x: (1/(sigma * math.sqrt(2*math.pi))) * math.exp((-1/2) * (((x-mu)/(sigma))**2))

#Future Implements

def model(k):
    params = (mass, k, spring_length, sd_tie, rope_length, space_h)
    s, l, p = sim.spring_rope_mass_system(*params)
    return cost(s.bounce_height_stochastic(time_step))

#Var Inf Setup

def prior():
    return lambda c: (1/(100 * torch.sqrt(torch.tensor(2*math.pi)))) * torch.exp((-1/2) * (((c-50)/(100))**2))

def joint():
    def breakout(z, x):
        j = list(map(prior(),z))
        def diff (c, z):
            constant = (1 / (10 * torch.sqrt(torch.tensor(2*math.pi))))
            temp = z[0]
            in_f = (-1 / 2) * (((c - temp) / (10)) ** 2)
            non_zero_f = torch.exp(in_f)
            return constant * non_zero_f + torch.tensor(10**(-8))
        k = []
        for i in x:
            t = diff(i, z)
            k.append(t)
        last = j[0] * torch.prod(torch.tensor(k)) + torch.tensor(10**(-8))
        return last
    return breakout

def guide():
    return lambda k, mu, sigma: (1/(sigma * torch.sqrt(torch.tensor(2*math.pi)))) * torch.exp((-1/2) * (((k[0]-mu)/(sigma))**2)) + 10**(-8)

def ELBO_loss(j, g, m_p, obs):
    mu, sigma = m_p
    s_points = np.linspace(-5, 5, 1000)
    def elbo_exp(c,m,s):
        j_c = j()([m+c[0]*s], obs)
        q_c = g()([m+c[0]*s], mu, sigma)
        return torch.log(j_c) - torch.log(q_c)
    expectation_exp = lambda x: elbo_exp((x,), mu, sigma) * torch.tensor(normalpdf(0,1)(x))
    eval_at = [expectation_exp(i) for i in s_points]
    dx = torch.tensor(10/len(eval_at))
    int_result = 0
    for i in range(len(eval_at)-1):
        int_result += (eval_at[i] + eval_at[i+1])/(2) * dx
    return int_result

def learn(samples, itrs, gamma):
    mu = torch.tensor(1000.0, requires_grad=True)
    sigma = torch.tensor(100.0, requires_grad=True)
    for _ in trange(itrs):
        l = ELBO_loss(joint, guide, (mu, sigma), samples)
        l.backward()
        dmu = mu.grad
        dsig = sigma.grad
        print(f"Loss: {l}, dmu: {dmu}, dsigma: {dsig}, mu: {mu}, sigma {sigma}")
        with torch.no_grad():
            mu -= gamma * dmu
        mu.grad.zero_()
        sigma.grad.zero_()

if __name__ == "__main__":
    s = list(generate_samples(10))
    learn(s, 5, 100)