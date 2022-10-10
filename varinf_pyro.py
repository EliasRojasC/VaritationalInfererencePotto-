import sim
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import DCTAdam
import pyro.distributions as dist

mass = 1
time_step = 0.1
spring_length = 1
rope_length = 1
sd_tie = 0.08
space_h = 10
k_real = 32

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

def model(data):
    k = pyro.sample("latent_spring_constant", dist.Normal(torch.tensor(100), torch.tensor(50)))
    for i in range(len(data)):
        pyro.sample("obs_{}".format(i), cost(k), obs=data[i])


def guide(data):
    s_q = pyro.param("sigma", torch.tensor(1.0), constraint=constraints.positive)
    m_q = pyro.param("mu", torch.tensor(50.0))
    pyro.sample("latent_spring_const", dist.Normal(s_q, m_q))



if __name__ == "__main__":
    # set up the optimizer
    adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
    optimizer = DCTAdam(adam_params)

    # setup the inference algorithm
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    data = [torch.tensor(cost(i)) for i in generate_samples(10)]

    n_steps = 5000
    # do gradient steps
    for step in range(n_steps):
        svi.step(data)

    sigma = pyro.param("sigma").item()
    mu = pyro.param("mu").item()