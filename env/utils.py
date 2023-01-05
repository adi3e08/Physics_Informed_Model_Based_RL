import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np

def create_background(width, height):
    background = pygame.Surface((width, height))
    pygame.draw.rect(background, (0,0,0), pygame.Rect(0, 0, width, height))
    return background

def rect_points(center, width, height, ang, scaling, offset):
    points = []
    diag = np.sqrt(width**2+height**2)/2
    ang1 = 2*np.arctan2(height,width)
    ang2 = 2*np.arctan2(width,height)
    # Pygame's y axis points downwards. Hence invert y coordinate alone before offset -> 
    points.append((offset[0]+scaling*(center[0]+np.cos(ang+ang1/2)*diag),\
                   offset[1]-scaling*(center[1]+np.sin(ang+ang1/2)*diag)))   
    points.append((offset[0]+scaling*(center[0]+np.cos(ang+ang1/2+ang2)*diag),\
                   offset[1]-scaling*(center[1]+np.sin(ang+ang1/2+ang2)*diag)))
    points.append((offset[0]+scaling*(center[0]+np.cos(ang+ang1*1.5+ang2)*diag),\
                   offset[1]-scaling*(center[1]+np.sin(ang+ang1*1.5+ang2)*diag)))
    points.append((offset[0]+scaling*(center[0]+np.cos(ang+ang1*1.5+2*ang2)*diag),\
                   offset[1]-scaling*(center[1]+np.sin(ang+ang1*1.5+2*ang2)*diag)))
    
    return points

def wrap(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def post_process(u_t, env_name):    
    if env_name == "pendulum":
        return np.array([wrap(u_t[0]),u_t[1]])
    
    elif env_name == "reacher" or env_name == "acrobot":
        return np.array([wrap(u_t[0]),wrap(u_t[1]),u_t[2],u_t[3]])
    
    elif env_name == "cartpole":
        return np.array([u_t[0],wrap(u_t[1]),u_t[2],u_t[3]])
    
    elif env_name == "cart2pole":
        return np.array([u_t[0],wrap(u_t[1]),wrap(u_t[2]),u_t[3],u_t[4],u_t[5]])

    elif env_name == "cart3pole":
        return np.array([u_t[0],wrap(u_t[1]),wrap(u_t[2]),wrap(u_t[3]),u_t[4],u_t[5],u_t[6],u_t[7]])

    elif env_name == "acro3bot":
        return np.array([wrap(u_t[0]),wrap(u_t[1]),wrap(u_t[2]),u_t[3],u_t[4],u_t[5]])

def make_env(name, mle=False):
    
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
        
    env = Env(mle)

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

def energy_check(name, seed):
    import matplotlib.pyplot as plt
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    np.random.seed(seed)
    env = make_env(name)
    times = []
    energies = []
    t = 0
    _, _, _ = env.reset()
    env.set_state(0.01*np.random.randn(2*env.n))
    
    o_t = env.get_obs()
    ep_r = 0
    while True:
        times.append(env.t)
        energies.append(env.calculate_total_energy())
        # env.render()
        a_t = np.zeros(env.action_size)
        o_t_1, r_t, done = env.step(a_t)
        ep_r += r_t
        t += 1
        o_t = o_t_1
        if done:
            print("Episode finished with total reward ",ep_r,"time steps",t)
            break
    times = np.array(times)
    energies = np.array(energies)
    delta_energies = ((energies-energies[0])/energies[0])*100

    # Plot Delta Energy vs Time
    plt.plot(times,delta_energies,'blue')
    plt.title(name.capitalize())
    plt.ylabel('% Change in Energy')
    plt.xlabel('Time steps')
    path = './env/energy_check'
    if os.path.exists(path):
        pass            
    else:
        os.makedirs(path)
    plt.savefig(os.path.join(path,env.name+'.png'), bbox_inches='tight')   
    plt.close()

