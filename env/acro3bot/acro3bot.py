import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from env import rewards
from env.utils import rect_points, wrap, basic_check
from env.base import BaseEnv

class acro3bot(BaseEnv):
    def __init__(self, mle):
        m1 = 0.1
        self.l1 = 1
        self.r1 = self.l1/2
        I1 = m1 * self.l1**2 / 12
        m2 = 0.1
        self.l2 = 1
        self.r2 = self.l2/2
        I2 = m2 * self.l2**2 / 12
        m3 = 0.1
        self.l3 = 1
        self.r3 = self.l3/2
        I3 = m3 * self.l3**2 / 12
        g = 9.8
        super(acro3bot, self).__init__(name = "acro3bot",
                                       n = 3,
                                       obs_size = 9,
                                       action_size = 2,
                                       inertials = [m1,self.l1,self.r1,I1,m2,self.l2,self.r2,I2,
                                                    m3,self.l3,self.r3,I3,g],
                                       dt = 0.01,
                                       a_scale = np.array([0.5,2.0]),
                                       mle = mle)

    def wrap_state(self):
        self.state[:3] = wrap(self.state[:3])

    def reset_state(self):
        self.state = np.array([np.pi + 0.01*np.random.randn(),\
                               0.01*np.random.randn(),\
                               0.01*np.random.randn(),\
                               0,0,0])
        self.wrap_state()

    def get_obs(self):
        return np.array([np.cos(self.state[0]),np.sin(self.state[0]),\
                        np.cos(self.state[1]),np.sin(self.state[1]),\
                        np.cos(self.state[2]),np.sin(self.state[2]),\
                        self.state[3],\
                        self.state[4],\
                        self.state[5]
                        ])

    def get_reward(self):
        upright = (np.array([np.cos(self.state[0]), np.cos(self.state[0]+self.state[1]), np.cos(self.state[0]+self.state[1]+self.state[2])])+1)/2  
        
        qdot = self.state[self.n:]
        ang_vel = np.array([qdot[0],qdot[0]+qdot[1],qdot[0]+qdot[1]+qdot[2]])
        small_velocity = rewards.tolerance(ang_vel, margin=self.ang_vel_limit).min()
        small_velocity = (1 + small_velocity) / 2
        
        reward = upright.mean() * small_velocity

        self.reward_breakup.append([upright.mean(),small_velocity])

        return reward

    def get_power(self, a, sdot):
        return np.array([a[0]*sdot[0]+a[1]*sdot[2]])

    def draw(self):
        offset = [250, 250]
        scaling = 75
        height = 0.125
        
        joint2 = [offset[0],offset[1]]
        link2_center = [(self.l1/2)*np.sin(self.state[0]),\
                       (self.l1/2)*np.cos(self.state[0])]
        link2_points = rect_points(link2_center, self.l1, height, np.pi/2-self.state[0],scaling,offset)
        pygame.draw.polygon(self.screen, (72,209,204), link2_points) # medium turquoise
        pygame.draw.circle(self.screen,(255,69,0), joint2, scaling*height/1.8) # orange red

        joint3 = [offset[0]+scaling*((self.l1)*np.sin(self.state[0])),offset[1]-scaling*((self.l1)*np.cos(self.state[0]))]
        link3_center = np.array([(self.l1)*np.sin(self.state[0])+(self.l2/2)*np.sin(self.state[0]+self.state[1]),\
                                  (self.l1)*np.cos(self.state[0])+(self.l2/2)*np.cos(self.state[0]+self.state[1])])
        link3_points = rect_points(link3_center, self.l2, height, np.pi/2-self.state[0]-self.state[1],scaling,offset)
        pygame.draw.polygon(self.screen, (72,209,204), link3_points) # medium turquoise
        pygame.draw.circle(self.screen,(255,69,0), joint3, scaling*height/1.8) # orange red

        joint4 = [offset[0]+scaling*((self.l1)*np.sin(self.state[0])+(self.l2)*np.sin(self.state[0]+self.state[1])),\
                  offset[1]-scaling*((self.l1)*np.cos(self.state[0])+(self.l2)*np.cos(self.state[0]+self.state[1]))]
        link4_center = np.array([(self.l1)*np.sin(self.state[0])+(self.l2)*np.sin(self.state[0]+self.state[1])\
                                 +(self.l3/2)*np.sin(self.state[0]+self.state[1]+self.state[2]),\
                                 (self.l1)*np.cos(self.state[0])+(self.l2)*np.cos(self.state[0]+self.state[1])\
                                 +(self.l3/2)*np.cos(self.state[0]+self.state[1]+self.state[2])])
        link4_points = rect_points(link4_center, self.l3, height, np.pi/2-self.state[0]-self.state[1]-self.state[2],scaling,offset)
        pygame.draw.polygon(self.screen, (72,209,204), link4_points) # medium turquoise
        pygame.draw.circle(self.screen,(255,69,0), joint4, scaling*height/1.8) # orange red

        end_point_plot = np.array([(self.l1)*np.sin(self.state[0])+(self.l2)*np.sin(self.state[0]+self.state[1])\
                                 +(self.l3)*np.sin(self.state[0]+self.state[1]+self.state[2]),\
                                 (self.l1)*np.cos(self.state[0])+(self.l2)*np.cos(self.state[0]+self.state[1])\
                                 +(self.l3)*np.cos(self.state[0]+self.state[1]+self.state[2])])
        end_point = [offset[0]+scaling*end_point_plot[0],offset[1]-scaling*end_point_plot[1]]
        # pygame.draw.circle(self.screen,(255,69,0), end_point, scaling*height/1.8) # orange red #endpoint

if __name__ == '__main__':
    basic_check("acro3bot",0)
