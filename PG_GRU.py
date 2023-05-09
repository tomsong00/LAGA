import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque



# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()

class PGNetwork(nn.Module):
    #GRU
    def __init__(self, state_dim,action_dim,hidden_size=1,num_layers=1):
        super(PGNetwork, self).__init__()
        self.gru = nn.GRU(state_dim, hidden_size, num_layers)  # utilize the GRU model in torch.nn
        self.linear1 = nn.Linear(hidden_size, 20)  # 全连接层
        self.linear2 = nn.Linear(20, action_dim)  # 全连接层
        self.softmax = nn.Softmax(dim=1)


    def forward(self, _x):
        x, hn = self.gru(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(s, b, -1)
        x = self.softmax(x)

        return x

class PG(object):
    # dqn Agent
    def __init__(self, inDim,outDim):  # 初始化
        # 状态空间和动作空间的维度，需要修改
        self.state_dim = inDim
        self.action_dim = outDim
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):
        observation = torch.FloatTensor(observation).to(device)
        network_output = self.network.forward(observation)

        if use_gpu:
            prob_weights=network_output.cpu().detach().numpy()
        # prob_weights = F.softmax(network_output, dim=0).detach().numpy()
        #概率随机选择
        #action = np.random.choice(range(prob_weights.shape[0]),p=prob_weights)  # select action w.r.t the actions prob
        action=np.max(prob_weights,axis=0)
        action=np.max(action,axis=0)
        #action=np.argmax(action)
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        #收益值
        self.ep_rs.append(r)

    def learn(self):
        self.time_step += 1

        # Step 1: 计算每一步的状态价值
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            #每次循环，把上一个G值乘以折扣(gamma) ，然后加上这个state获得的reward即可。我们把这个值记录在discounted_ep_rs
            running_add = running_add * GAMMA + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        discounted_ep_rs=discounted_ep_rs.astype(float)
        if np.sum(discounted_ep_rs)!=0:
            # 归一化处理
            discounted_ep_rs -= np.mean(discounted_ep_rs)  # 减均值
            discounted_ep_rs /= np.std(discounted_ep_rs)  # 除以标准差
            discounted_ep_rs = torch.FloatTensor(discounted_ep_rs).to(device)

        # Step 2: 前向传播
        np_ep_obs=np.array(self.ep_obs)
        torch_ep_obs=torch.FloatTensor(np_ep_obs)
        softmax_input = self.network.forward(torch_ep_obs.to(device))

        if use_gpu:
            softmax_input=softmax_input.cpu().detach().numpy()
        softmax_input=np.max(softmax_input,axis=1)
        softmax_input =torch.FloatTensor(softmax_input).to(device)

        #all_act_prob = F.softmax(softmax_input, dim=0)
        neg_log_prob = F.cross_entropy(input=softmax_input, target=torch.LongTensor(self.ep_as).to(device),
                                       reduction='none')
        # Step 3: 反向传播
        #在原来的差距乘以G值，也就是以G值作为更新
        loss = torch.mean(neg_log_prob * discounted_ep_rs)
        self.optimizer.zero_grad()
        #1、通过网络，求出预测值pre的分布。
        # 2、和真实值action进行比较，求得neg_log_prob
        # 3、最终求得neg_log_prob乘以G值，求得loss
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()
        #完成以上更新为thera t+1
        print(loss)
        # 每次学习完后清空数组
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []






