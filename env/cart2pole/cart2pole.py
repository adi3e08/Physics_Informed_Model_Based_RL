import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from env import rewards
from env.utils import rect_points, wrap, basic_check, pygame_transform
from env.base import BaseEnv

class cart2pole(BaseEnv):
    def __init__(self):
        m1 = 1
        l1, r1, I1 = 0, 0, 0 # dummy
        
        m2 = 0.1
        l2 = 1
        r2 = l2/2
        I2 = m2 * l2**2 / 12
        
        m3 = 0.1
        l3 = 1
        r3 = l3/2
        I3 = m3 * l3**2 / 12
        
        g = 9.8

        m = [m1,m2,m3]
        l = [l1,l2,l3]
        r = [r1,r2,r3]
        I = [I1,I2,I3]
        super(cart2pole, self).__init__(name = "cart2pole",
                                        n = 3,
                                        obs_size = 8,
                                        action_size = 1,
                                        inertials = m+l+r+I+[g],
                                        a_scale = np.array([10.0]))
    def wrap_state(self):
        self.state[1:3] = wrap(self.state[1:3])

    def reset_state(self):
        self.state = np.array([0.01*np.random.randn(),
                               np.pi + 0.01*np.random.randn(),
                               0.01*np.random.randn(),
                               0,0,0])

    def get_A(self, a):
        a_1, = np.clip(a, -1.0, 1.0)*self.a_scale
        a_2, a_3 = 0.0, 0.0
        return np.array([a_1,a_2,a_3])

    def get_obs(self):
        return np.array([self.state[0],
                        np.cos(self.state[1]),np.sin(self.state[1]),
                        np.cos(self.state[2]),np.sin(self.state[2]),
                        self.state[3],
                        self.state[4],
                        self.state[5]
                        ])

    def get_reward(self):
        upright = (np.array([np.cos(self.state[1]), np.cos(self.state[1]+self.state[2])])+1)/2

        if np.abs(self.state[0]) <= self.x_limit:
            centered = rewards.tolerance(self.state[0], margin=self.x_limit)
            centered = (1 + centered) / 2
        else:
            centered = 0.1

        qdot = self.state[self.n:]
        ang_vel = np.array([qdot[0],qdot[1],qdot[1]+qdot[2]])       
        small_velocity = rewards.tolerance(ang_vel[1:], margin=self.ang_vel_limit).min()
        small_velocity = (1 + small_velocity) / 2
        
        reward = upright.mean() * small_velocity * centered

        return reward

    def draw(self):        
        centers, joints, angles = self.geo

        #horizontal rail
        pygame.draw.polygon(self.screen, self.rail_color, rect_points([0,0], self.rail_length, self.rail_width, np.pi/2, self.scaling, self.offset)) 

        plot_x = ((centers[0,0] + self.x_limit) % (2 * self.x_limit)) - self.x_limit
        link1_points = rect_points([plot_x,0], self.cart_length, self.cart_width, np.pi/2, self.scaling, self.offset) 
        pygame.draw.polygon(self.screen, self.cart_color, link1_points)

        for j in range(1,self.n):
            link_points = rect_points(centers[j], self.link_length, self.link_width, angles[j,0],self.scaling,self.offset)
            pygame.draw.polygon(self.screen, self.link_color, link_points)
        
            joint_point = pygame_transform(joints[j],self.scaling,self.offset)
            pygame.draw.circle(self.screen, self.joint_color, joint_point, self.scaling*self.joint_radius)

if __name__ == '__main__':
    basic_check("cart2pole",0)