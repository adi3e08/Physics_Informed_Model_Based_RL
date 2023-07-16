import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import time
import dill as pickle
pickle.settings['recurse'] = True
import pygame
import numpy as np
from env.utils import create_background
from copy import deepcopy

class BaseEnv:
    def __init__(self,name,n,obs_size,action_size,inertials,a_scale):
        self.name = name
        self.n = n
        self.obs_size = obs_size
        self.action_size = action_size
        self.inertials = inertials
        self.a_scale = a_scale

        self.dt = 0.01 # time per simulation step (in seconds)
        self.t = 0 # elapsed simulation steps
        self.t_max = 1000  # max simulation steps
        self.state = np.zeros(self.n)
        self.ang_vel_limit = 20.0

        with open("./env/"+self.name+"/robot.p", "rb") as inf:
            funcs = pickle.load(inf)                
        self.kinematics = funcs['kinematics']
        self.dynamics = funcs['dynamics']

        # For rendering
        self.display = False
        self.screen_width = 500
        self.screen_height = 500
        self.offset = [250, 250]
        self.scaling = 75
        self.x_limit = 2.0

        self.link_length = 1.0
        self.link_width = 0.2
        self.link_color = (72,209,204) # medium turquoise

        self.joint_radius = self.link_width/1.8
        self.joint_color = (205,55,0) # orange red

        self.cart_length = 5*self.link_width
        self.cart_width = 2*self.link_width
        self.cart_color = (200,255,0) # yellow

        self.rail_length = 2*self.x_limit
        self.rail_width = self.link_width/2.5
        self.rail_color = (150,150,150) # gray

    def wrap_state(self):
        pass

    def reset_state(self):
        pass

    def get_A(self, a):
        pass

    def get_obs(self):
        pass

    def get_reward(self):
        pass

    def draw(self):
        pass

    def set_state(self,s):
        self.state = s

    def reset(self):
        self.reset_state()
        self.wrap_state()
        self.geo = self.kinematics(self.inertials+self.state.tolist())
        self.t = 0

        return self.get_obs(), 0.0, False

    def step(self, a):
        self.state = self.rk4(self.state, self.get_A(a))
        self.wrap_state()
        self.geo = self.kinematics(self.inertials+self.state.tolist())
        
        self.t += 1
        if self.t >= self.t_max: # infinite horizon formulation, no terminal state, similar to dm_control
            done = True
        else:
            done = False

        return self.get_obs(), self.get_reward(), done

    def F(self, s, a):
        M, C, G = self.dynamics(self.inertials+s.tolist())
        qdot = s[self.n:]
        qddot = np.linalg.inv(M+1e-6*np.eye(self.n)) @ (a - C @ qdot - G.flatten()) 
        
        return np.concatenate((qdot,qddot))

    def rk4(self, s, a):
        s0 = deepcopy(s)
        k = []
        for l in range(4):
            if l > 0:
                if l == 1 or l == 2:
                    dt = self.dt/2
                elif l == 3:
                    dt = self.dt
                s = s0 + dt * k[l-1]
            k.append(self.F(s, a))
        s = s0 + (self.dt/6.0) * (k[0] + 2 * k[1] + 2 * k[2] + k[3])
        
        return s

    def render(self):
        if self.display:
            self.screen.blit(self.background, (0, 0))
            self.draw()
            time.sleep(0.006)
            pygame.display.flip()
        else:
            self.display = True
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption(self.name)
            self.background = create_background(self.screen_width, self.screen_height)
