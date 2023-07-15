import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from env import rewards
from env.utils import rect_points, wrap, basic_check, pygame_transform
from env.base import BaseEnv

class reacher(BaseEnv):
    def __init__(self):
        m1 = 0.1
        l1 = 1
        r1 = l1/2
        I1 = m1 * l1**2 / 12
        
        m2 = 0.1
        l2 = 1
        r2 = l2/2
        I2 = m2 * l2**2 / 12
        
        g = 0.0
        
        m = [m1,m2]
        l = [l1,l2]
        r = [r1,r2]
        I = [I1,I2]
        super(reacher, self).__init__(name = "reacher",
                                      n = 2,
                                      obs_size = 6,
                                      action_size = 2,
                                      inertials = m+l+r+I+[g],
                                      a_scale = np.array([0.1,0.1]))
        self.dt = 0.02
        self.goal_position = np.array([0,2])

    def wrap_state(self):
        self.state[:2] = wrap(self.state[:2])

    def reset_state(self):
        self.state = np.array([np.pi + 0.01*np.random.randn(),
                               0.01*np.random.randn(),
                               0,
                               0])

    def get_A(self, a):
        a_1, a_2 = np.clip(a, -1.0, 1.0)*self.a_scale
        return np.array([a_1,a_2])

    def get_obs(self):
        return np.array([np.cos(self.state[0]),np.sin(self.state[0]),
                        np.cos(self.state[1]),np.sin(self.state[1]),
                        self.state[2],
                        self.state[3]
                        ])

    def get_reward(self):
        upright = (np.array([np.cos(self.state[0]), np.cos(self.state[0]+self.state[1])])+1)/2

        qdot = self.state[self.n:]
        ang_vel = np.array([qdot[0],qdot[0]+qdot[1]])        
        small_velocity = rewards.tolerance(ang_vel, margin=self.ang_vel_limit).min()
        small_velocity = (1 + small_velocity) / 2
        
        reward = upright.mean() * small_velocity

        return reward

    def draw(self):
        pygame.draw.circle(self.screen,'slateblue1', [self.offset[0]+self.scaling*self.goal_position[0],
                                                      self.offset[1]-self.scaling*self.goal_position[1]], self.scaling*self.link_width)

        centers, joints, angles = self.geo

        for j in range(self.n):
            link_points = rect_points(centers[j], self.link_length, self.link_width, angles[j,0],self.scaling,self.offset)
            pygame.draw.polygon(self.screen, self.link_color, link_points)
        
            joint_point = pygame_transform(joints[j],self.scaling,self.offset)
            pygame.draw.circle(self.screen, self.joint_color, joint_point, self.scaling*self.joint_radius)


if __name__ == '__main__':
    basic_check("reacher",0)