import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np

def create_background(length, width):
    background = pygame.Surface((length, width))
    pygame.draw.rect(background, (0,0,0), pygame.Rect(0, 0, length, width))
    return background

def rect_points(center, length, width, ang, scaling, offset):
    points = []
    diag = np.sqrt(length**2+width**2)/2
    ang1 = 2*np.arctan2(width,length)
    ang2 = 2*np.arctan2(length,width)

    points.append((center[0]+np.sin(ang+ang1/2)*diag, center[1]+np.cos(ang+ang1/2)*diag))   
    
    points.append((center[0]+np.sin(ang+ang1/2+ang2)*diag, center[1]+np.cos(ang+ang1/2+ang2)*diag))
    
    points.append((center[0]+np.sin(ang+ang1*1.5+ang2)*diag, center[1]+np.cos(ang+ang1*1.5+ang2)*diag))
    
    points.append((center[0]+np.sin(ang+ang1*1.5+2*ang2)*diag, center[1]+np.cos(ang+ang1*1.5+2*ang2)*diag))
    
    return [pygame_transform(point, scaling, offset) for point in points]

def pygame_transform(point, scaling, offset):
    # Pygame's y axis points downwards. Hence invert y coordinate alone before offset.
    return (offset[0]+scaling*point[0],offset[1]-scaling*point[1]) 

def wrap(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def make_env(name):
    
    if name == "pendulum":
        from env.pendulum.pendulum import pendulum as Env

    elif name == "reacher":
        from env.reacher.reacher import reacher as Env
    
    elif name == "cartpole":
        from env.cartpole.cartpole import cartpole as Env

    elif name == "acrobot":
        from env.acrobot.acrobot import acrobot as Env
        
    elif name == "cart2pole":
        from env.cart2pole.cart2pole import cart2pole as Env

    elif name == "acro3bot":
        from env.acro3bot.acro3bot import acro3bot as Env

    elif name == "cart3pole":
        from env.cart3pole.cart3pole import cart3pole as Env
        
    env = Env()

    return env

def basic_check(name, seed):
    np.random.seed(seed)
    env = make_env(name)
    no_episodes = 1
    for episode in range(no_episodes):
        t = 0
        o_t, _, _ = env.reset()
        ep_r = 0
        while True:
            env.render()
            a_t = np.random.uniform(-1,1,env.action_size)
            o_t_1, r_t, done = env.step(a_t)
            ep_r += r_t
            t += 1
            o_t = o_t_1
            if done:
                print("Episode finished with total reward ",ep_r,"time steps",t)
                break


