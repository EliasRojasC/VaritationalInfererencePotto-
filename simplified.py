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
        return 100


def generate_samples(count):
    params = (mass, k_real, spring_length, sd_tie, rope_length, space_h)
    sys = sim.spring_rope_mass_system(*params)
    for _ in range(count):
        s, l, p = sys.bounce_height_stochastic(0.001)
        yield s




if __name__ == "__main__":
    s = list(generate_samples(10))
    #learn(s, 5, 100)