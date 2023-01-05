import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from env import rewards
from env.utils import rect_points, wrap, basic_check
from env.base import BaseEnv

class cartpole(BaseEnv):
    def __init__(self, mle):
        m1 = 1
        m2 = 0.1
        self.l2 = 1
        self.r2 = self.l2/2
        I2 = m2 * self.l2**2 / 12
        g = 9.8
        super(cartpole, self).__init__(name = "cartpole",
                                       n = 2,
                                       obs_size = 5,
                                       action_size = 1,
                                       inertials = [m1,m2,self.l2,self.r2,I2,g],
                                       dt = 0.01,
                                       a_scale = np.array([10.0]),
                                       mle = mle)

    def wrap_state(self):
        self.state[1] = wrap(self.state[1])

    def reset_state(self):
        self.state = np.array([0.01*np.random.randn(),\
                               np.pi + 0.01*np.random.randn(),\
                               0,0])
        self.wrap_state()

    def get_obs(self):
        return np.array([self.state[0],\
                        np.cos(self.state[1]),np.sin(self.state[1]),\
                        self.state[2],\
                        self.state[3]\
                        ])

    def get_reward(self):
        upright = (np.array([np.cos(self.state[1])])+1)/2

        if np.abs(self.state[0]) <= self.x_limit:
            centered = rewards.tolerance(self.state[0], margin=self.x_limit)
            centered = (1 + centered) / 2
        else:
            centered = 0.1

        qdot = self.state[self.n:]
        ang_vel = qdot
        small_velocity = rewards.tolerance(ang_vel[1:], margin=self.ang_vel_limit).min()
        small_velocity = (1 + small_velocity) / 2

        reward = upright.mean() * small_velocity * centered

        self.reward_breakup.append([upright.mean(),small_velocity,centered])

        return reward

    def get_power(self, a, sdot):
        return np.array([a[0]*sdot[0]])       
    
    def draw(self):
        offset = [250, 250]
        scaling = 75
        height = 0.15
        
        x_limit = self.x_limit
        plot_x = ((self.state[0] + x_limit) % (2 * x_limit)) - x_limit
        # plot_x = self.state[0]
        
        link1_center = [plot_x,0]
        link1_points = rect_points(link1_center, 5*height, 2*height, 0,scaling,offset) 
        
        joint2 = [offset[0]+scaling*plot_x,offset[1]]
        link2_center = [plot_x+(self.l2/2)*np.sin(self.state[1]),\
                       (self.l2/2)*np.cos(self.state[1])]
        link2_points = rect_points(link2_center, self.l2, height, np.pi/2-self.state[1],scaling,offset)
        
        pygame.draw.polygon(self.screen, (150,150,150), rect_points([0,0], 2*x_limit, height/3, 0, scaling,offset)) # gray horizontal rail

        pygame.draw.polygon(self.screen, (200,255,0), link1_points) # yellow
        pygame.draw.polygon(self.screen, (72,209,204), link2_points) # medium turquoise
        pygame.draw.circle(self.screen,(255,69,0), joint2, scaling*height/1.8) # orange red

if __name__ == '__main__':
    basic_check("cartpole",0)
