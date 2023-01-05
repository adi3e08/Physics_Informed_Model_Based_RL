import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import time
import dill as pickle
pickle.settings['recurse'] = True
import pygame
import numpy as np
from scipy.integrate import solve_ivp
from env import rewards
from env.utils import create_background, rect_points, wrap, basic_check, post_process
from copy import deepcopy

class BaseEnv:
    def __init__(self,name,n,obs_size,action_size,inertials,dt,a_scale,mle):
        self.name = name
        self.n = n
        self.obs_size = obs_size
        self.action_size = action_size
        self.inertials = inertials
        self.dt = dt # time per simulation step
        self.a_scale = a_scale
        self.mle = mle

        self.t = 0 # no of simulation steps
        self.t_max = 1000  # max simulation steps
        self.state = np.zeros(self.n)
        self.w = np.array([0.0]) # work done by non conservative forces (motor torques)

        with open("./env/"+self.name+"/dynamics.p", "rb") as inf:
            funcs = pickle.load(inf)                
        self.H = funcs['H']
        self.F = funcs['F']
        if self.mle:
            self.jac = funcs['D']
            self.phi_dim = 2*self.n
            self.phi = np.identity(self.phi_dim)
            self.phi_len = self.phi_dim*self.phi_dim
        self.a_zeros = np.zeros(self.action_size)

        self.display = False
        self.screen_width = 500
        self.screen_height = 500
        self.x_limit = 2.0
        self.ang_vel_limit = 20.0

    def wrap_state(self):
        pass

    def reset_state(self):
        pass

    def get_obs(self):
        pass

    def get_reward(self):
        pass

    def get_power(self, a, sdot):
        pass

    def draw(self):
        pass

    def set_state(self,s):
        self.state = s

    def calculate_total_energy(self):
        H = self.H(self.inertials+self.state.tolist())
        return H
        
    def get_jac(self, a):
        return self.jac(self.inertials+self.state.tolist()+a.tolist())

    def get_components(self, s_all):
        if self.mle:
            s = s_all[:-(self.phi_len+1+self.action_size)]
            a = s_all[-(self.phi_len+1+self.action_size):-(self.phi_len+1)]
            w = s_all[-(self.phi_len+1):-self.phi_len]
            phi = s_all[-self.phi_len:].reshape((self.phi_dim,self.phi_dim))
            return s,a,w,phi
        else:
            s = s_all[:-(1+self.action_size)]
            a = s_all[-(1+self.action_size):-1]
            w = s_all[-1:]
            return s,a,w

    def rk4(self, s_all, da_ds):
        s_all_orig = deepcopy(s_all)
        k = []
        da = np.zeros((self.action_size,self.phi_dim))
        for l in range(4):
            s,a,w,phi = self.get_components(s_all)
            sdot = self.F(self.inertials+s.tolist()+a.tolist()).flatten()
            jac_s, jac_a = self.jac(self.inertials+s.tolist()+a.tolist())
            phidot = np.zeros((self.phi_dim,self.phi_dim))
            for m in range(self.phi_dim):
                phi[:,m] = post_process(phi[:,m],self.name)
                if l == 0:
                    da[:,m] = np.matmul(da_ds,phi[:,m])
                phidot[:,m] = (np.matmul(jac_s,phi[:,m]) + np.matmul(jac_a,da[:,m]))
            k_l = np.concatenate((sdot, self.a_zeros, self.get_power(a, sdot), phidot.flatten()))
            k.append(k_l)
            if l == 0 or l == 1:
                s_all = (s_all_orig + self.dt * k_l / 2)
            elif l == 2:
                s_all = (s_all_orig + self.dt * k_l)
            elif l == 3:
                s_all = s_all_orig + self.dt / 6.0 * (k[0] + 2 * k[1] + 2 * k[2] + k[3])
        
        return s_all

    def reset(self):
        self.reset_state()

        if self.mle:
            self.phi = np.identity(self.phi_dim)
                
        self.t = 0
        self.w = np.array([0.0])

        self.reward_breakup = []

        return self.get_obs(), 0.0, False 

    def step(self, a, da_ds=None):
        s = self.state
        a = np.clip(a, -1.0, 1.0)
        w = self.w
        
        if self.mle:
            s_all = np.concatenate((s, a*self.a_scale, w, self.phi.flatten()))
            ns = self.rk4(s_all,np.array([da_ds[j]*self.a_scale[j] for j in range(self.action_size)]))
            ns, nw, nphi = ns[:-(self.phi_len+1+self.action_size)], ns[-(self.phi_len+1):-self.phi_len], ns[-self.phi_len:]  # omit action
            self.phi = nphi.reshape((self.phi_dim,self.phi_dim))
            for m in range(self.phi_dim):
                self.phi[:,m] = post_process(self.phi[:,m],self.name)
        else:
            s_all = np.concatenate((s, a*self.a_scale, w))
            ns = solve_ivp(self._dsdt, [0, self.dt], s_all, method='DOP853')
            ns = ns.y[:,-1]  # only care about final timestep
            ns, nw = ns[:-(1+self.action_size)], ns[-1:]  # omit action

        self.w = nw
        self.state = ns
        self.wrap_state()
        self.t += 1
                        
        if self.t >= self.t_max: # infinite horizon formulation, no terminal state, similar to dm_control
            done = True
        else:
            done = False

        return self.get_obs(), self.get_reward(), done

    def _dsdt(self, t, s_all):
        s,a,w = self.get_components(s_all)
        sdot = self.F(self.inertials+s.tolist()+a.tolist()).flatten()
        return np.concatenate((sdot, self.a_zeros, self.get_power(a, sdot)))

    def render(self):
        if self.display:
            self.screen.blit(self.background, (0, 0))
            self.draw()
            time.sleep(0.01)
            pygame.display.flip()
        else:
            self.display = True
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption(self.name)
            self.background = create_background(self.screen_width, self.screen_height)

if __name__ == '__main__':
    basic_check("cartpole",2,False)
