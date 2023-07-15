import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.func import jacrev
from collections import deque
import random

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, o, a, r, o_1):            
        self.buffer.append((o, a, r, o_1))
    
    def sample_transitions(self, batch_size):
        O, A, R, O_1 = zip(*random.sample(self.buffer, batch_size))
        return torch.stack(O), torch.stack(A), torch.stack(R), torch.stack(O_1)

    def sample_states(self, batch_size):
        O, A, R, O_1 = zip(*random.sample(self.buffer, batch_size))
        return torch.stack(O)

    def __len__(self):
        return len(self.buffer)

# Critic network
class V_FC(torch.nn.Module):
    def __init__(self, obs_size):
        super(V_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 1)

    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2).view(-1)        
        return y

# Actor network
class Pi_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Pi_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.mu = torch.nn.Linear(256, action_size)
        self.log_sigma = torch.nn.Linear(256, action_size)

    def forward(self, x, deterministic=False, with_logprob=False):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        mu = self.mu(y2)

        if deterministic:
            # used for evaluating policy
            action = torch.tanh(mu)
            log_prob = None
        else:
            log_sigma = self.log_sigma(y2)
            log_sigma = torch.clamp(log_sigma,min=-20.0,max=2.0)
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            x_t = dist.rsample()
            action = torch.tanh(x_t)
            if with_logprob:
                log_prob = dist.log_prob(x_t).sum(1)
                log_prob -= torch.log(torch.clamp(1-action.pow(2),min=1e-6)).sum(1)
            else:
                log_prob = None

        return action, log_prob


# DNN dynamics model
class dnn(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(dnn, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size+action_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, obs_size)

    def forward(self, x, a):
        y1 = F.relu(self.fc1(torch.cat((x,a),1)))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2)        
        return y

# LNN dynamics model
class lnn(torch.nn.Module):
    def __init__(self, env_name, n, obs_size, action_size, dt, a_zeros):
        super(lnn, self).__init__()
        self.env_name = env_name
        self.dt = dt
        self.n = n

        input_size = obs_size - self.n
        out_L = int(self.n*(self.n+1)/2)
        self.fc1_L = torch.nn.Linear(input_size, 64)
        self.fc2_L = torch.nn.Linear(64, 64)
        self.fc3_L = torch.nn.Linear(64, out_L)
        if not self.env_name == "reacher":
            self.fc1_V = torch.nn.Linear(input_size, 64)
            self.fc2_V = torch.nn.Linear(64, 64)
            self.fc3_V = torch.nn.Linear(64, 1)

        self.a_zeros = a_zeros

    def trig_transform_q(self, q):
        if self.env_name == "pendulum":
            return torch.column_stack((torch.cos(q[:,0]),torch.sin(q[:,0])))
        
        elif self.env_name == "reacher" or self.env_name == "acrobot":
            return torch.column_stack((torch.cos(q[:,0]),torch.sin(q[:,0]),\
                                       torch.cos(q[:,1]),torch.sin(q[:,1])))
        
        elif self.env_name == "cartpole":
            return torch.column_stack((q[:,0],\
                                       torch.cos(q[:,1]),torch.sin(q[:,1])))
        
        elif self.env_name == "cart2pole":
            return torch.column_stack((q[:,0],\
                                       torch.cos(q[:,1]),torch.sin(q[:,1]),\
                                       torch.cos(q[:,2]),torch.sin(q[:,2])))

        elif self.env_name == "cart3pole":
            return torch.column_stack((q[:,0],\
                                       torch.cos(q[:,1]),torch.sin(q[:,1]),\
                                       torch.cos(q[:,2]),torch.sin(q[:,2]),\
                                       torch.cos(q[:,3]),torch.sin(q[:,3])))
        
        elif self.env_name == "acro3bot":
            return torch.column_stack((torch.cos(q[:,0]),torch.sin(q[:,0]),\
                                       torch.cos(q[:,1]),torch.sin(q[:,1]),\
                                       torch.cos(q[:,2]),torch.sin(q[:,2])))

    def inverse_trig_transform_model(self, x):
        if self.env_name == "pendulum":
            return torch.cat((torch.atan2(x[:,1],x[:,0]).unsqueeze(1),x[:,2:]),1)
        
        elif self.env_name == "reacher" or self.env_name == "acrobot":
            return torch.cat((torch.atan2(x[:,1],x[:,0]).unsqueeze(1),torch.atan2(x[:,3],x[:,2]).unsqueeze(1),x[:,4:]),1)
        
        elif self.env_name == "cartpole":
            return torch.cat((x[:,0].unsqueeze(1),torch.atan2(x[:,2],x[:,1]).unsqueeze(1),x[:,3:]),1)
        
        elif self.env_name == "cart2pole":
            return torch.cat((x[:,0].unsqueeze(1),torch.atan2(x[:,2],x[:,1]).unsqueeze(1),torch.atan2(x[:,4],x[:,3]).unsqueeze(1),x[:,5:]),1)

        elif self.env_name == "cart3pole":
            return torch.cat((x[:,0].unsqueeze(1),torch.atan2(x[:,2],x[:,1]).unsqueeze(1),torch.atan2(x[:,4],x[:,3]).unsqueeze(1),
                              torch.atan2(x[:,6],x[:,5]).unsqueeze(1),x[:,7:]),1)

        elif self.env_name == "acro3bot":
            return torch.cat((torch.atan2(x[:,1],x[:,0]).unsqueeze(1),torch.atan2(x[:,3],x[:,2]).unsqueeze(1),torch.atan2(x[:,5],x[:,4]).unsqueeze(1),
                              x[:,6:]),1)

    def compute_L(self, q):
        y1_L = F.softplus(self.fc1_L(q))
        y2_L = F.softplus(self.fc2_L(y1_L))
        y_L = self.fc3_L(y2_L)
        device = y_L.device
        if self.n == 1:
            L = y_L.unsqueeze(1)

        elif self.n == 2:
            L11 = y_L[:,0].unsqueeze(1)
            L1_zeros = torch.zeros(L11.size(0),1, dtype=torch.float64, device=device)

            L21 = y_L[:,1].unsqueeze(1)
            L22 = y_L[:,2].unsqueeze(1)

            L1 = torch.cat((L11,L1_zeros),1) 
            L2 = torch.cat((L21,L22),1)
            L = torch.cat((L1.unsqueeze(1),L2.unsqueeze(1)),1)
        
        elif self.n == 3:
            L11 = y_L[:,0].unsqueeze(1)
            L1_zeros = torch.zeros(L11.size(0),2, dtype=torch.float64, device=device)

            L21 = y_L[:,1].unsqueeze(1)
            L22 = y_L[:,2].unsqueeze(1)
            L2_zeros = torch.zeros(L21.size(0),1, dtype=torch.float64, device=device)

            L31 = y_L[:,3].unsqueeze(1)
            L32 = y_L[:,4].unsqueeze(1)
            L33 = y_L[:,5].unsqueeze(1)

            L1 = torch.cat((L11,L1_zeros),1) 
            L2 = torch.cat((L21,L22,L2_zeros),1)
            L3 = torch.cat((L31,L32,L33),1)
            L = torch.cat((L1.unsqueeze(1),L2.unsqueeze(1),L3.unsqueeze(1)),1)
        
        elif self.n == 4:
            L11 = y_L[:,0].unsqueeze(1)
            L1_zeros = torch.zeros(L11.size(0),3, dtype=torch.float64, device=device)

            L21 = y_L[:,1].unsqueeze(1)
            L22 = y_L[:,2].unsqueeze(1)
            L2_zeros = torch.zeros(L21.size(0),2, dtype=torch.float64, device=device)

            L31 = y_L[:,3].unsqueeze(1)
            L32 = y_L[:,4].unsqueeze(1)
            L33 = y_L[:,5].unsqueeze(1)
            L3_zeros = torch.zeros(L31.size(0),1, dtype=torch.float64, device=device)

            L41 = y_L[:,6].unsqueeze(1)
            L42 = y_L[:,7].unsqueeze(1)
            L43 = y_L[:,8].unsqueeze(1)
            L44 = y_L[:,9].unsqueeze(1)

            L1 = torch.cat((L11,L1_zeros),1) 
            L2 = torch.cat((L21,L22,L2_zeros),1)
            L3 = torch.cat((L31,L32,L33,L3_zeros),1)
            L4 = torch.cat((L41,L42,L43,L44),1)
            L = torch.cat((L1.unsqueeze(1),L2.unsqueeze(1),L3.unsqueeze(1),L4.unsqueeze(1)),1)

        return L

    def get_A(self, a):
        if self.env_name == "pendulum" or self.env_name == "reacher":
            A = a
        
        elif self.env_name == "acrobot":
            A = torch.cat((self.a_zeros,a),1)
        
        elif self.env_name == "cartpole" or self.env_name == "cart2pole":
            A = torch.cat((a,self.a_zeros),1)
        
        elif self.env_name == "cart3pole" or self.env_name == "acro3bot":
            A = torch.cat((a[:,:1],self.a_zeros,a[:,1:]),1)

        return A

    def get_L(self, q):
        trig_q = self.trig_transform_q(q)
        L = self.compute_L(trig_q)         
        return L.sum(0), L

    def get_V(self, q):
        trig_q = self.trig_transform_q(q)
        y1_V = F.softplus(self.fc1_V(trig_q))
        y2_V = F.softplus(self.fc2_V(y1_V))
        V = self.fc3_V(y2_V).squeeze()
        return V.sum()

    def get_acc(self, q, qdot, a):
        dL_dq, L = jacrev(self.get_L, has_aux=True)(q)
        term_1 = torch.einsum('blk,bijk->bijl', L, dL_dq.permute(2,3,0,1))
        dM_dq = term_1 + term_1.transpose(2,3)
        c = torch.einsum('bjik,bk,bj->bi', dM_dq, qdot, qdot) - 0.5 * torch.einsum('bikj,bk,bj->bi', dM_dq, qdot, qdot)        
        Minv = torch.cholesky_inverse(L)
        dV_dq = 0 if self.env_name == "reacher" else jacrev(self.get_V)(q)
        qddot = torch.matmul(Minv,(self.get_A(a)-c-dV_dq).unsqueeze(2)).squeeze(2)
        return qddot                                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                           
    def derivs(self, s, a):
        q, qdot = s[:,:self.n], s[:,self.n:]
        qddot = self.get_acc(q, qdot, a)
        return torch.cat((qdot,qddot),dim=1)                                                                                                                                                               

    def rk2(self, s, a):                                                                                                                                                                                   
        alpha = 2.0/3.0 # Ralston's method                                                                                                                                                                 
        k1 = self.derivs(s, a)                                                                                                                                                                             
        k2 = self.derivs(s + alpha * self.dt * k1, a)                                                                                                                                                      
        s_1 = s + self.dt * ((1.0 - 1.0/(2.0*alpha))*k1 + (1.0/(2.0*alpha))*k2)                                                                                                                            
        return s_1

    def forward(self, o, a):
        s_1 = self.rk2(self.inverse_trig_transform_model(o), a)
        o_1 = torch.cat((self.trig_transform_q(s_1[:,:self.n]),s_1[:,self.n:]),1)
        return o_1

# Reward model
class reward_model_FC(torch.nn.Module):
    def __init__(self, obs_size):
        super(reward_model_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2).view(-1)   
        return y
