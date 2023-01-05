import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from env import rewards
from env.utils import rect_points, wrap, basic_check
from env.base import BaseEnv

class pendulum(BaseEnv):
    def __init__(self, mle):
        m1 = 1
        self.l1 = 1
        self.r1 = self.l1
        I1 = m1 * self.l1**2
        g = 9.8
        super(pendulum, self).__init__(name = "pendulum",
                                       n = 1,
                                       obs_size = 3,
                                       action_size = 1,
                                       inertials = [m1,self.l1,self.r1,I1,g],
                                       dt = 0.02,
                                       a_scale = np.array([2.0]),
                                       mle = mle)

    def wrap_state(self):
        self.state[0] = wrap(self.state[0])

    def reset_state(self):
        self.state = np.array([np.pi+0.01*np.random.randn(),0])
        self.wrap_state()

    def get_obs(self):
        return np.array([np.cos(self.state[0]),np.sin(self.state[0]),\
                        self.state[1]
                        ])

    def get_reward(self):
        upright = (np.array([np.cos(self.state[0])])+1)/2

        qdot = self.state[self.n:]
        ang_vel = qdot
        small_velocity = rewards.tolerance(ang_vel, margin=self.ang_vel_limit).min()
        small_velocity = (1 + small_velocity) / 2

        reward = upright.mean() * small_velocity
        self.reward_breakup.append([upright.mean(),small_velocity])

        return reward

    def get_power(self, a, sdot):
        return np.array([a[0]*sdot[0]])       
    
    def draw(self):
        offset = [250, 250]
        scaling = 75
        height = 0.1
        
        joint1 = [offset[0],offset[1]]
        link1_center = [(self.l1/2)*np.sin(self.state[0]),\
                       (self.l1/2)*np.cos(self.state[0])]
        link1_points = rect_points(link1_center, self.l1, height/1.8, np.pi/2-self.state[0],scaling,offset)
        pygame.draw.polygon(self.screen, (72,209,204), link1_points) # medium turquoise
        pygame.draw.circle(self.screen,(255,69,0), joint1, scaling*height/1.25) # orange red

        end_point = np.array([(self.l1)*np.sin(self.state[0]),(self.l1)*np.cos(self.state[0])])
        end_point_plot = [offset[0]+scaling*end_point[0],offset[1]-scaling*end_point[1]]
        pygame.draw.circle(self.screen,'slateblue1', end_point_plot, scaling*0.15) # orange red #endpoint

if __name__ == '__main__':
    basic_check("pendulum",0)
