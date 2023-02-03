import math
from numpy.random import normal
from matplotlib import pyplot as plt, animation as an
from celluloid import Camera
import torch

class spring_rope_mass_system:
    def __init__(self, mass, k, spring_l, rope_s, rope_l, height):
        self.m = mass
        self.k = k if k>= 0 else 0
        self.s_l = spring_l
        self.r_s = rope_s
        self.h = height
        self.r_l = rope_l

class srms_pure_python(spring_rope_mass_system):
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


class srms_torch(spring_rope_mass_system):
    def bounce_height_deterministic(self, time_step, mu, sigma):
        g = 9.8
        x_t = self.h - (self.s_l + self.r_l)
        if x_t <= 0: return self.h
        xdot_t = -g*math.sqrt((x_t)/g)
        while xdot_t < 0:
            x_next = x_t + time_step*xdot_t
            xdot_next = xdot_t + time_step*((self.k/self.m)*(self.h-self.s_l-self.r_l-x_t) - g)
            x_t = x_next
            xdot_t = xdot_next
        x_torch = torch.tensor(x_t)
        out = (1 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(
            (-1 / 2) * ((((self.h - self.r_l - x_torch) - mu) / (sigma)) ** 2))
        return out

    def bounce_height_deterministic_graph(self, time_step, mu, sigma):
        s, p, v = self.bounce_height_deterministic(time_step)
        plt.plot(p, 'go')
        plt.show()



class spring_rope_teg_sim:
    def __init__(self, mass, k_1, k_2, rope_l_1, height):
        self.m = mass
        self.k_1 = k_1 if k_1 > 0 else 0
        self.k_2 = k_2 if k_2 > 0 else 0
        self.rope_l_1 = rope_l_1
        self.height = height

    def bounce_height_deterministic(self, time_step):
        g = 9.8
        s_1_pos = self.height
        s_2_pos = self.height
        s_1_posdot = 0
        s_2_posdot = 0
        t = 0
        postrack_2 = []
        postrack_1 = []
        while s_2_posdot < 0 or t == 0:
            s_2_pos_next = s_2_pos + time_step * s_2_posdot
            s_2_posdot_next = s_2_posdot + time_step * ((self.k_2/self.m) * (s_1_pos - s_2_pos) - g)
            s_1_pos_next = s_1_pos + time_step * s_1_posdot if (self.height - s_1_pos + time_step * s_1_posdot) < self.rope_l_1 else self.rope_l_1
            spring_contr_b = -(self.k_2/self.m) * (s_1_pos - s_2_pos)
            spring_contr_a = (self.k_1) * (self.height - s_1_pos) if self.height - s_1_pos < self.rope_l_1 else -(self.k_1) * (self.rope_l_1)
            s_1_posdot_next = s_1_posdot + time_step * (spring_contr_b + spring_contr_a)
            s_1_pos = s_1_pos_next
            s_2_pos = s_2_pos_next
            s_1_posdot = s_1_posdot_next
            s_2_posdot = s_2_posdot_next
            t += 1
            postrack_2.append(s_2_pos)
            postrack_1.append(s_1_pos)
        return s_2_pos, postrack_2, postrack_1



if __name__ == '__main__':
    mass = 1
    time_step = 0.1
    spring_length = 5
    rope_length = 0
    sd_tie = 0.08
    space_h = 10
    k_real = 32
    sim = srms_pure_python(mass, k_real, spring_length, 0, rope_length, space_h)
    end, pos, vel = sim.bounce_height_stochastic(0.001)

    fig = plt.figure()
    board = plt.axes(xlim=(0, 2), ylim=(0, 12))
    camera = Camera(fig)

    for i in range(len(pos)//10):
            ones = [1]
            tr_1 = [pos[i*10]]
            plt.plot(ones, tr_1, marker="o", markersize=20, color="blue")
            camera.snap()

    print(len(pos))
    animation = camera.animate()
    animation.save('animation.mp4')



