import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from env import rewards
from env.utils import rect_points, wrap, basic_check
from env.base import BaseEnv

class reacher(BaseEnv):
    def __init__(self, mle):
        m1 = 0.1
        self.l1 = 1
        self.r1 = self.l1/2
        I1 = m1 * self.l1**2 / 12
        m2 = 0.1
        self.l2 = 1
        self.r2 = self.l2/2
        I2 = m2 * self.l2**2 / 12
        g = 0.0
        super(reacher, self).__init__(name = "reacher",
                                       n = 2,
                                       obs_size = 6,
                                       action_size = 2,
                                       inertials = [m1,self.l1,self.r1,I1,m2,self.l2,self.r2,I2,g],
                                       dt = 0.02,
                                       a_scale = np.array([0.1,0.1]),
                                       mle = mle)

    def wrap_state(self):
        self.state[:2] = wrap(self.state[:2])

    def reset_state(self):
        self.state = np.array([np.pi + 0.01*np.random.randn(),\
                               0.01*np.random.randn(),\
                               0,\
                               0])
        self.wrap_state()

    def get_obs(self):
        return np.array([np.cos(self.state[0]),np.sin(self.state[0]),\
                        np.cos(self.state[1]),np.sin(self.state[1]),\
                        self.state[2],\
                        self.state[3]\
                        ])

    def get_reward(self):
        upright = (np.array([np.cos(self.state[0]), np.cos(self.state[0]+self.state[1])])+1)/2

        qdot = self.state[self.n:]
        ang_vel = np.array([qdot[0],qdot[0]+qdot[1]])        
        small_velocity = rewards.tolerance(ang_vel, margin=self.ang_vel_limit).min()
        small_velocity = (1 + small_velocity) / 2
        
        reward = upright.mean() * small_velocity

        self.reward_breakup.append([upright.mean(),small_velocity])

        return reward

    def get_power(self, a, sdot):
        return np.array([a[0]*sdot[0]+a[1]*sdot[1]])

    def draw(self):
        offset = [250, 250]
        scaling = 75
        height = 0.15 

        dis_threshold = 0.15
        theta1  = 0
        theta2  = 0
        goal_position = np.array([self.l1*np.sin(theta1)+self.l2*np.sin(theta1+theta2),\
                                  self.l1*np.cos(theta1)+self.l2*np.cos(theta1+theta2)])
        pygame.draw.circle(self.screen,(0,255,0), [offset[0]+scaling*goal_position[0],\
                                                   offset[1]-scaling*goal_position[1]], scaling*dis_threshold)

        joint1 = offset
        link1_center = [(self.l1/2)*np.sin(self.state[0]),\
                      (self.l1/2)*np.cos(self.state[0])]
        link1_points = rect_points(link1_center, self.l1, height, np.pi/2-self.state[0],scaling,offset) 
        joint2 = [offset[0]+scaling*(self.l1)*np.sin(self.state[0]),\
                       offset[1]-scaling*(self.l1)*np.cos(self.state[0])]
        link2_center = [(self.l1)*np.sin(self.state[0])+\
                      (self.l2/2)*np.sin(self.state[0]+self.state[1]),\
                      (self.l1)*np.cos(self.state[0])+\
                      (self.l2/2)*np.cos(self.state[0]+self.state[1])]
        link2_points = rect_points(link2_center, self.l2, height, np.pi/2-self.state[0]-self.state[1],scaling,offset)

        pygame.draw.polygon(self.screen, (72,209,204), link1_points) # medium turquoise
        pygame.draw.circle(self.screen,(255,69,0), joint1, scaling*height/1.8) # orange red
        pygame.draw.polygon(self.screen, (72,209,204), link2_points) # medium turquoise
        pygame.draw.circle(self.screen,(255,69,0), joint2, scaling*height/1.8) # orange red

        end_point = [offset[0]+scaling*((self.l1)*np.sin(self.state[0])+\
                      (self.l2)*np.sin(self.state[0]+self.state[1])),\
                      offset[1]-scaling*((self.l1)*np.cos(self.state[0])+\
                      (self.l2)*np.cos(self.state[0]+self.state[1]))]
        pygame.draw.circle(self.screen,'slateblue1', end_point, scaling*height/1.8) # orange red 

if __name__ == '__main__':
    basic_check("reacher",0)