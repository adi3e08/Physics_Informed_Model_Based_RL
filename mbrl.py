import os
import argparse
from copy import deepcopy
import random
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from torch.utils.tensorboard import SummaryWriter
from env.utils import make_env
from models.mbrl import ReplayBuffer, V_FC, Pi_FC, dnn, lnn, reward_model_FC

# Model-Based RL algorithm
class MBRL:
    def __init__(self, arglist):
        self.arglist = arglist
        
        random.seed(self.arglist.seed)
        np.random.seed(self.arglist.seed)
        torch.manual_seed(self.arglist.seed)
        
        self.env = make_env(self.arglist.env)
        
        self.device = torch.device("cpu")

        path = "./log/"+self.env.name+"/mbrl_"+self.arglist.model
        self.exp_dir = os.path.join(path, "seed_"+str(self.arglist.seed))
        self.model_dir = os.path.join(self.exp_dir, "models")
        self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")

        self.actor = Pi_FC(self.env.obs_size,self.env.action_size).to(self.device)

        if self.arglist.mode == "train":
            self.critic = V_FC(self.env.obs_size).to(self.device)
            self.critic_target = deepcopy(self.critic)
            
            if self.arglist.model == "lnn":
                if self.env.action_size < self.env.n:
                    a_zeros = torch.zeros(self.arglist.batch_size,self.env.n-self.env.action_size, dtype=torch.float64, device=self.device)
                else:
                    a_zeros = None
                self.transition_model = lnn(self.env.name, self.env.n, self.env.obs_size, self.env.action_size, self.env.dt, a_zeros).to(self.device)
            
            elif self.arglist.model == "dnn":
                self.transition_model = dnn(self.env.obs_size, self.env.action_size).to(self.device)
            self.transition_loss_fn = torch.nn.L1Loss()

            self.reward_model = reward_model_FC(self.env.obs_size).to(self.device)
            self.reward_loss_fn = torch.nn.L1Loss()

            if self.arglist.resume:
                checkpoint = torch.load(os.path.join(self.model_dir,"emergency.ckpt"))
                self.start_episode = checkpoint['episode'] + 1

                self.actor.load_state_dict(checkpoint['actor'])
                self.critic.load_state_dict(checkpoint['critic'])
                self.critic_target.load_state_dict(checkpoint['critic_target'])
                self.transition_model.load_state_dict(checkpoint['transition_model'])
                self.reward_model.load_state_dict(checkpoint['reward_model'])

                self.replay_buffer = checkpoint['replay_buffer']

            else: 
                self.start_episode = 0

                self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)

                if os.path.exists(path):
                    pass            
                else:
                    os.makedirs(path)
                os.mkdir(self.exp_dir)
                os.mkdir(os.path.join(self.tensorboard_dir))
                os.mkdir(self.model_dir)

            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.arglist.lr)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.arglist.lr)
            self.transition_optimizer = torch.optim.AdamW(self.transition_model.parameters(), lr=self.arglist.lr)
            self.reward_optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=self.arglist.lr)

            if self.arglist.resume:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                self.transition_optimizer.load_state_dict(checkpoint['transition_optimizer'])
                self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer'])

                print("Done loading checkpoint ...")

            self.a_scale = torch.tensor(self.env.a_scale,dtype=torch.float64,device=self.device)

            self.train()

        elif self.arglist.mode == "eval":
            checkpoint = torch.load(self.arglist.checkpoint,map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            ep_r_list = self.eval(self.arglist.episodes,self.arglist.render)

    def save_checkpoint(self, name):
        checkpoint = {'actor' : self.actor.state_dict(),\
                      'critic' : self.critic.state_dict(),\
                     'transition_model' : self.transition_model.state_dict(),\
                     'reward_model' : self.reward_model.state_dict()
                     }
        torch.save(checkpoint, os.path.join(self.model_dir, name))

    def save_emergency_checkpoint(self, episode):
        checkpoint = {'episode' : episode,\
                      'actor' : self.actor.state_dict(),\
                      'actor_optimizer': self.actor_optimizer.state_dict(),\
                      'critic' : self.critic.state_dict(),\
                      'critic_optimizer': self.critic_optimizer.state_dict(),\
                      'critic_target' : self.critic_target.state_dict(),\
                     'transition_model' : self.transition_model.state_dict(),\
                      'transition_optimizer': self.transition_optimizer.state_dict(),\
                     'reward_model' : self.reward_model.state_dict(),\
                      'reward_optimizer': self.reward_optimizer.state_dict(),\
                      'replay_buffer' : self.replay_buffer \
                      }
        torch.save(checkpoint, os.path.join(self.model_dir, "emergency.ckpt"))

    def hard_update(self, target, source):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

    def train(self):
        critic_target_updates = 0

        writer = SummaryWriter(log_dir=self.tensorboard_dir)

        if not self.arglist.resume:
            # Initialize replay buffer with K random episodes
            for episode in range(self.arglist.K):
                o,_,_ = self.env.reset()
                o_tensor = torch.tensor(o, dtype=torch.float64, device=self.device)
                ep_r = 0
                while True:
                    a = np.random.uniform(-1.0, 1.0, size=self.env.action_size)
                    o_1,r,done = self.env.step(a)
                    a_tensor = torch.tensor(a, dtype=torch.float64, device=self.device)
                    o_1_tensor = torch.tensor(o_1, dtype=torch.float64, device=self.device)
                    r_tensor = torch.tensor(r, dtype=torch.float64, device=self.device)
                    self.replay_buffer.push(o_tensor, a_tensor, r_tensor, o_1_tensor)
                    ep_r += r
                    o_tensor = o_1_tensor
                    if done:
                        break
            
            print("Done initialization ...")
        print("Started training ...")

        for episode in range(self.start_episode,self.arglist.episodes):
            # Model learning
            transition_loss_list, reward_loss_list = [], []
            transition_grad_list, reward_grad_list = [], []
            for model_batches in range(self.arglist.model_batches):
                O, A, R, O_1 = self.replay_buffer.sample_transitions(self.arglist.batch_size)
                                
                # Dynamics learning
                O_1_pred = self.transition_model(O,A*self.a_scale)
                transition_loss = self.transition_loss_fn(O_1_pred, O_1)
                self.transition_optimizer.zero_grad()
                transition_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transition_model.parameters(), self.arglist.clip_term)
                self.transition_optimizer.step()
                transition_loss_list.append(transition_loss.item())
                transition_grad = []
                for param in self.transition_model.parameters():
                    if param.grad is not None:
                        transition_grad.append(param.grad.flatten())
                transition_grad_list.append(torch.norm(torch.cat(transition_grad)).item())

                # Reward learning                        
                R_pred = self.reward_model(O_1)
                reward_loss = self.reward_loss_fn(R_pred,R)
                self.reward_optimizer.zero_grad()
                reward_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.arglist.clip_term)
                self.reward_optimizer.step()
                reward_loss_list.append(reward_loss.item())
                reward_grad_list.append(torch.norm(torch.cat([param.grad.flatten() for param in self.reward_model.parameters()])).item())

            writer.add_scalar('transition_loss', np.mean(transition_loss_list), episode)
            writer.add_scalar('reward_loss', np.mean(reward_loss_list), episode)
            writer.add_scalar('transition_grad',np.mean(transition_grad_list),episode)
            writer.add_scalar('reward_grad',np.mean(reward_grad_list),episode)

            #Behaviour learning
            actor_loss_list, critic_loss_list = [], []
            actor_grad_list, critic_grad_list = [], []

            nan_count = 0
            for behaviour_batches in range(self.arglist.behaviour_batches):
                O = self.replay_buffer.sample_states(self.arglist.batch_size)
                t = 0
                values, values_target, values_lambda, R = [], [], [], []
                log_probs = []
                try:
                    while True:                                            
                        A, log_prob = self.actor(O, False, True)
                        log_probs.append(log_prob)
                        O_1 = self.transition_model(O, A*self.a_scale)
                        R.append(self.reward_model(O_1))
                        values.append(self.critic(O))
                        values_target.append(self.critic_target(O))                  
                        t += 1
                        O = O_1
                        if t % self.arglist.T == 0:
                            values_target.append(self.critic_target(O_1))
                            break

                    # lambda-return calculation
                    gae = torch.zeros_like(R[0])                                               
                    for t_ in reversed(range(self.arglist.T)):
                        delta = R[t_]+self.arglist.gamma*values_target[t_+1]-values_target[t_]
                        gae = delta+self.arglist.gamma*self.arglist.Lambda*gae
                        values_lambda.append(gae+values_target[t_])
                    values_lambda = torch.stack(values_lambda)
                    values_lambda = values_lambda.flip(0)

                    values = torch.stack(values)
                    critic_loss = 0.5*torch.pow(values-values_lambda.detach(),2).sum(0).mean()

                    log_probs = torch.stack(log_probs)
                    actor_loss = - (values_lambda-0.0001*log_probs).sum(0).mean()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward(inputs=[param for param in self.critic.parameters()])
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.arglist.clip_term)

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward(inputs=[param for param in self.actor.parameters()])
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.arglist.clip_term)

                    critic_grad = torch.norm(torch.cat([param.grad.flatten() for param in self.critic.parameters()]))
                    actor_grad = torch.norm(torch.cat([param.grad.flatten() for param in self.actor.parameters()]))

                    if torch.isnan(critic_grad).any().item() or torch.isnan(actor_grad).any().item():
                        nan_count += 1
                    else:
                        self.critic_optimizer.step()
                        self.actor_optimizer.step()

                        critic_target_updates = (critic_target_updates+1)%100
                        if critic_target_updates == 0:
                            self.hard_update(self.critic_target, self.critic)

                        actor_loss_list.append(actor_loss.item())
                        critic_loss_list.append(critic_loss.item())
                        actor_grad_list.append(actor_grad.item())
                        critic_grad_list.append(critic_grad.item())

                except:
                    nan_count += 1
            
            if nan_count > 0:
                print("episode",episode,"got nan during behaviour learning","nan count",nan_count)
            writer.add_scalar('critic_loss',np.mean(critic_loss_list),episode)
            writer.add_scalar('actor_loss',np.mean(actor_loss_list),episode)
            writer.add_scalar('critic_grad',np.mean(critic_grad_list),episode)
            writer.add_scalar('actor_grad',np.mean(actor_grad_list),episode)

            # Environment Interaction
            o,_,_ = self.env.reset()
            o_tensor = torch.tensor(o, dtype=torch.float64, device=self.device).unsqueeze(0)
            ep_r = 0
            while True:
                with torch.no_grad():
                    try:
                        a_tensor, _ = self.actor(o_tensor)
                    except:
                        print("episode",episode,"got nan during environment interaction")
                        break
                o_1,r,done = self.env.step(a_tensor.cpu().numpy()[0])
                o_1_tensor = torch.tensor(o_1, dtype=torch.float64, device=self.device).unsqueeze(0)
                r_tensor = torch.tensor(r, dtype=torch.float64, device=self.device)
                self.replay_buffer.push(o_tensor[0], a_tensor[0], r_tensor, o_1_tensor[0])
                ep_r += r
                o_tensor = o_1_tensor
                if done:
                    writer.add_scalar('ep_r', ep_r, episode)
                    if episode % self.arglist.eval_every == 0 or episode == self.arglist.episodes-1:
                        try:
                            # Evaluate agent performance
                            eval_ep_r_list = self.eval(self.arglist.eval_over)
                            writer.add_scalar('eval_ep_r', np.mean(eval_ep_r_list), episode)
                            self.save_checkpoint(str(episode)+".ckpt")
                        except:
                            print("episode",episode,"got nan during eval")
                    if (episode % 25 == 0 or episode == self.arglist.episodes-1) and episode > self.start_episode:
                        self.save_emergency_checkpoint(episode)
                    break

    def eval(self, episodes, render=False):
        # Evaluate agent performance over several episodes
        ep_r_list = []
        for episode in range(episodes):
            o,_,_ = self.env.reset()
            ep_r = 0
            while True:
                with torch.no_grad():
                    a, _ = self.actor(torch.tensor(o, dtype=torch.float64, device=self.device).unsqueeze(0),True)
                a = a.cpu().numpy()[0]   
                o_1,r,done = self.env.step(a)
                if render:
                    self.env.render()
                ep_r += r
                o = o_1
                if done:
                    ep_r_list.append(ep_r)
                    if render:
                        print("Episode finished with total reward ",ep_r)
                    break
        
        if self.arglist.mode == "eval":
            print("Average return :",np.mean(ep_r_list))
        
        return ep_r_list    

def parse_args():
    parser = argparse.ArgumentParser("Model-Based Reinforcement Learning")
    # Common settings
    parser.add_argument("--env", type=str, default="acrobot", help="pendulum / reacher / cartpole / acrobot / cart2pole / acro3bot / cart3pole")
    parser.add_argument("--mode", type=str, default="train", help="train or eval")
    parser.add_argument("--episodes", type=int, default=500, help="number of episodes to run experiment for")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    # Core training parameters
    parser.add_argument("--resume", action="store_true", default=False, help="continue training from checkpoint")
    parser.add_argument("--model", type=str, default="lnn", help="lnn / dnn")
    parser.add_argument("--T", type=int, default=16, help="imagination horizon")
    parser.add_argument("--K", type=int, default=10, help="init replay buffer with K random episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--clip-term", type=float, default=100, help="gradient clipping norm")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--Lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for model learning, behaviour learning")
    parser.add_argument("--model-batches", type=int, default=int(1e4), help="model batches per episode")
    parser.add_argument("--behaviour-batches", type=int, default=int(1e3), help="behaviour batches per episode")
    parser.add_argument("--replay-size", type=int, default=int(1e5), help="replay buffer size")
    parser.add_argument("--eval-every", type=int, default=5, help="eval every _ episodes during training")
    parser.add_argument("--eval-over", type=int, default=50, help="each time eval over _ episodes")
    # Eval settings
    parser.add_argument("--checkpoint", type=str, default="", help="path to checkpoint")
    parser.add_argument("--render", action="store_true", default=False, help="render")
    return parser.parse_args()

if __name__ == '__main__':
    arglist = parse_args()
    mbrl = MBRL(arglist)
