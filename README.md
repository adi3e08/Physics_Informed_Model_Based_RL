# Physics-Informed Model-Based RL

This work has been accepted at <a href="https://l4dc.seas.upenn.edu/" target="_blank">Learning for Dynamics & Control Conference (L4DC) 2023</a>.

We apply model-based RL to robotic systems undergoing rigid body motion without contacts. 
<p align="center">
<img src="https://adi3e08.github.io/files/research/pimbrl/envs.png" width="100%"/>
</p>

In our model-based RL algorithm, we learn a model of the environment, essentially its transition dynamics and reward function, use it to generate imaginary trajectories and backpropagate through them to update the policy, exploiting the differentiability of the model. Intuitively, learning more accurate models should lead to better performance.

We compare two versions of our model-based RL algorithm, one which uses a standard deep neural network based dynamics model and the other which uses a much more accurate, physics-informed neural network based dynamics model.

We show that, in model-based RL, model accuracy mainly matters in environments that are sensitive to initial conditions. In these environments, the physics-informed version of our algorithm achieves significantly better average-return and sample efficiency. In environments that are not sensitive to initial conditions, both versions of our algorithm achieve similar average-return, while the physics-informed version achieves better sample efficiency. 

We measure the sensitivity to initial conditions by computing the rate of separation of trajectories which start from nearby initial states, i. e., by computing the Lyapunov exponents. The sensitivity to initial conditions depends on factors such as the system dynamics, control policy, degree of actuation and damping.

We also show that, in challenging environments, where we need a lot of samples to learn, physics-informed model-based RL can achieve better average-return than state-of-the-art model-free RL algorithms such as Soft Actor-Critic, by generating accurate imaginary data.

For more information check out,
- [Project Webpage](https://adi3e08.github.io/research/pimbrl)
- [Preprint](https://arxiv.org/abs/2212.02179)

## Requirements
- Python
- Numpy
- Pytorch
- Tensorboard
- Pygame

## Usage
To train MBRL LNN on Acrobot task, run,

    python mbrl.py --env acrobot --mode train --episodes 500 --seed 0 

The data from this experiment will be stored in the folder "./log/acrobot/mbrl_lnn/seed_0". This folder will contain two sub folders, (i) models : here model checkpoints will be stored and (ii) tensorboard : here tensorboard plots will be stored.

To evaluate MRBL LNN on Acrobot task, run,

    python mbrl.py --env acrobot --mode eval --episodes 3 --seed 100 --checkpoint ./log/acrobot/mbrl_lnn/seed_0/models/500.ckpt --render
