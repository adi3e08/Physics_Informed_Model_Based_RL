# Physics-Informed Model-Based RL

Published at <a href="https://l4dc.seas.upenn.edu/" target="_blank">Learning for Dynamics & Control Conference (L4DC), 2023</a>.

<p align="center">
<img src="https://adi3e08.github.io/files/research/pimbrl/envs.png" width="100%"/>
</p>

## Abstract
We apply reinforcement learning (RL) to robotics tasks. One of the drawbacks of traditional RL algorithms has been their poor sample efficiency. One approach to improve the sample efficiency is model-based RL. In our model-based RL algorithm, we learn a model of the environment, essentially its transition dynamics and reward function, use it to generate imaginary trajectories and backpropagate through them to update the policy, exploiting the differentiability of the model.

Intuitively, learning more accurate models should lead to better model-based RL performance. Recently, there has been growing interest in developing better deep neural network based dynamics models for physical systems, by utilizing the structure of the underlying physics. We focus on robotic systems undergoing rigid body motion without contacts. We compare two versions of our model-based RL algorithm, one which uses a standard deep neural network based dynamics model and the other which uses a much more accurate, physics-informed neural network based dynamics model. 

We show that, in model-based RL, model accuracy mainly matters in environments that are sensitive to initial conditions, where numerical errors accumulate fast. In these environments, the physics-informed version of our algorithm achieves significantly better average-return and sample efficiency. In environments that are not sensitive to initial conditions, both versions of our algorithm achieve similar average-return, while the physics-informed version achieves better sample efficiency. 

We also show that, in challenging environments, physics-informed model-based RL achieves better average-return than state-of-the-art model-free RL algorithms such as Soft Actor-Critic, as it computes the policy-gradient analytically.

For more information check out,
- [Project Webpage](https://adi3e08.github.io/research/pimbrl)
- [Paper](https://arxiv.org/abs/2212.02179)

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

    python mbrl.py --env acrobot --mode eval --episodes 3 --seed 100 --checkpoint ./log/acrobot/mbrl_lnn/seed_0/models/499.ckpt --render

## Citation
If you find this work helpful, please consider starring this repo and citing our paper using the following Bibtex.
```bibtex
@inproceedings{ramesh2023physics,
  title={Physics-Informed Model-Based Reinforcement Learning},
  author={Ramesh, Adithya and Ravindran, Balaraman},
  booktitle={Learning for Dynamics and Control Conference},
  pages={26--37},
  year={2023},
  organization={PMLR}
}


