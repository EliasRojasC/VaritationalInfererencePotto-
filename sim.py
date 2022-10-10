import math
from numpy.random import normal
import matplotlib.pyplot as plt

class spring_rope_mass_system:
    def __init__(self, mass, k, spring_l, rope_s, rope_l, height):
        self.m = mass
        self.k = k if k>= 0 else 0
        self.s_l = spring_l
        self.r_s = rope_s
        self.h = height
        self.r_l = rope_l

    def bounce_height_deterministic(self, time_step):
        g = 9.8
        x_t = self.h - (self.s_l + self.r_l)
        if x_t <= 0: return self.h
        xdot_t = -g*math.sqrt((x_t)/g)
        while xdot_t < 0:
            x_next = x_t + time_step*xdot_t
            xdot_next = xdot_t + time_step*((self.k/self.m)*(self.h-self.s_l-self.r_l-x_t) - g)
            x_t = x_next
            xdot_t = xdot_next
        return x_t

    def bounce_height_stochastic(self, time_step):
        g = 9.8
        stoc = normal(0, self.r_s, 1)[0]
        if stoc > 3*self.r_s: stoc = 3*self.r_s
        rp_s = (self.r_l - 3*self.r_s) + stoc
        x_t = self.h - (self.s_l + rp_s)
        if x_t <= 0: return self.h
        xdot_t = -g * math.sqrt((x_t) / g)
        l_p = [x_t]
        l_v = [xdot_t]
        while xdot_t < 0:
            x_next = x_t + time_step * xdot_t
            xdot_next = xdot_t + time_step * ((self.k / self.m) * (self.h - self.s_l - rp_s - x_t) - g)
            x_t = x_next
            xdot_t = xdot_next
            l_p.append(x_t)
            l_v.append(xdot_t)
        return x_t, l_p, l_v

    def bounce_height_stochastic_graph(self, time_step):
        s, p, v = self.bounce_height_stochastic(time_step)
        plt.plot(p, 'go')
        plt.show()

if __name__ == '__main__':
    system = spring_rope_mass_system(1, 70, 3, 0.08, 1, 10)
    system.bounce_height_stochastic_graph(0.001)

