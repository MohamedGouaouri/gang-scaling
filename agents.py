
from collections import namedtuple, deque
from torch_geometric.nn import GCNConv, global_mean_pool
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
GAMMA = 0.99
EPS_START = 0.5
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class ChainGraphDQN(nn.Module):

    def __init__(self, num_node_features, n_actions, n_microservices, edge_index, batch, device='cpu'):
        super(ChainGraphDQN, self).__init__()

        self.conv1 = GCNConv(num_node_features, 16)
        self.layer1 = nn.Linear(16, 64)
        self.layer2 = nn.Linear(64, 64)
        self.out_layers = nn.ModuleList([nn.Linear(64, n_actions) for _ in range(n_microservices)])

        self.edge_index = edge_index.to(device)
        self.batch = batch.to(device)

        self.device = device


    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, x):
        x = x.to(torch.float32).to(self.device)
        x = self.conv1(x, self.edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, self.batch)
        x = F.elu(self.layer1(x))
        x = F.elu(self.layer2(x))

        out = torch.stack([net(x) for net in self.out_layers], dim=1)
        return out
steps_done = 0

def select_action(env, policy_net, state):
    # print(state)
    global steps_done
    sample = random.random()
    # exponential decaying eps
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # print(f"State in select_action {state}")
            out = policy_net(state.unsqueeze(0))
            # print(f'Using policy net {out.argmax(1).shape}')
            return out.squeeze().argmax(1)
    else:
        out = torch.tensor([env.action_space.sample()], device=device)
        return out


def optimize_model(policy_net, target_net, optimizer, criterion, memory):

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])

    # return batch.action
    # print(batch.state)
    # print(batch.action)

    # print(len(batch.action), batch.action[0].shape)
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print(f"From replay {state_batch.shape}, {len(batch.state)} {len(batch.state[0])}")
    r = policy_net(state_batch).squeeze()
    # print(f"r shape: {r.shape}, action_batch shape: {action_batch.shape}")
    state_action_values = r.gather(2, action_batch.unsqueeze(2)).squeeze()


    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(2).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros((BATCH_SIZE, 4), device=device)
    # print(f"non_final_mask shape: {non_final_mask.shape}") # shape: [16]
    # print(target_net(non_final_next_states).max(2).values.shape) # shape: [16, 4, 10]
    # print(f"next_state_values shape {next_state_values.shape}") # shape: [16, 4]

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).squeeze().max(2).values
        # next_state_values[non_final_mask] = target_net(non_final_next_states).squeeze().squeeze().max(2)[0]

    # Compute the expected Q values

    # print(next_state_values.shape, reward_batch.shape, reward_batch.unsqueeze(1).shape)
    # print(reward_batch)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    # print(f"state_action_values shape: {state_action_values.shape} expected_state_action_values {expected_state_action_values.shape}", )
    loss = criterion(state_action_values.to(torch.float), expected_state_action_values.to(torch.float))
    # print(loss.item())
    # # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 10)
    optimizer.step()

    return loss.detach().item()