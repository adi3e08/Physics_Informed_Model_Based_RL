import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from env import rewards
from env.utils import rect_points, wrap, basic_check, pygame_transform
from env.base import BaseEnv

class pendulum(BaseEnv):
    def __init__(self):
        m1 = 1
        l1 = 1
        r1 = l1
        I1 = m1 * l1**2
        
        g = 9.8

        m = [m1]
        l = [l1]
        r = [r1]
        I = [I1]
        super(pendulum, self).__init__(name = "pendulum",
                                       n = 1,
                                       obs_size = 3,
                                       action_size = 1,
                                       inertials = m+l+r+I+[g],
                                       a_scale = np.array([2.0]))
        self.dt = 0.02

    def wrap_state(self):
        self.state[0] = wrap(self.state[0])

    def reset_state(self):
        self.state = np.array([np.pi+0.01*np.random.randn(),0])

    def get_A(self, a):
        return np.clip(a, -1.0, 1.0)*self.a_scale

    def get_obs(self):
        return np.array([np.cos(self.state[0]),np.sin(self.state[0]),
                        self.state[1]
                        ])

    def get_reward(self):
        upright = (np.array([np.cos(self.state[0])])+1)/2

        qdot = self.state[self.n:]
        ang_vel = qdot
        small_velocity = rewards.tolerance(ang_vel, margin=self.ang_vel_limit).min()
        small_velocity = (1 + small_velocity) / 2

        reward = upright.mean() * small_velocity

        return reward   
    
    def draw(self):
        centers, joints, angles = self.geo

        link1_center = (centers[0]+joints[0])/2
        link1_points = rect_points(link1_center, self.link_length, self.link_width/2.5, angles[0,0],self.scaling,self.offset)
        pygame.draw.polygon(self.screen, self.link_color, link1_points)

        for j in range(self.n):
            center_point = [self.offset[0]+self.scaling*centers[j,0],self.offset[1]-self.scaling*centers[j,1]]
            pygame.draw.circle(self.screen, 'slateblue1', center_point, self.scaling*self.joint_radius*1.5)
        
            joint_point = pygame_transform(joints[j],self.scaling,self.offset)
            pygame.draw.circle(self.screen, self.joint_color, joint_point, self.scaling*self.joint_radius)

if __name__ == '__main__':
    basic_check("pendulum",0)
