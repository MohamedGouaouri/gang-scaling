from deployment import Deployment
from chain import MicroServiceChain
from load_generator import RequestPath
from types_ import RequestLoadType
from collections import namedtuple, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np

import logging
import os
import warnings



# Create a directory to save models
os.makedirs("models", exist_ok=True)

# Set up logging
logging.basicConfig(filename='training.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

logging.getLogger('apscheduler.executors.default').setLevel(logging.ERROR)
logging.getLogger('apscheduler.executors.default').propagate = False

deps = [
    Deployment(
        name="d1",
        replicas=1,
        pod_cpu_limit=4000,
        pod_memory_limit=2**30, # 1 gb
        pod_arrival_rate=10,
        pod_service_rate=20,
        create=True,
        # lb_strategy=DeploymentLoadBalancingStrategy.RANDOM
    ),
    Deployment(
        name="d2",
        replicas=2,
        pod_cpu_limit=4000, # 4 cpus
        pod_memory_limit=2**30, # 1 gb
        pod_arrival_rate=5,
        pod_service_rate=4,
        create=True,
        # lb_strategy=DeploymentLoadBalancingStrategy.RANDOM
    ),
    Deployment(
        name="d3",
        replicas=3,
        pod_cpu_limit=1000, # 1 cpus
        pod_memory_limit=2**30, # 1 gb
        pod_arrival_rate=10,
        pod_service_rate=12,
        create=True,
        # lb_strategy=DeploymentLoadBalancingStrategy.RANDOM
    ),
    Deployment(
        name="d4",
        replicas=2,
        pod_cpu_limit=3000, # 3 cpus
        pod_memory_limit=2**30, # 1 gb
        pod_arrival_rate=5,
        pod_service_rate=10,
        create=True,
        # lb_strategy=DeploymentLoadBalancingStrategy.RANDOM

    )
]

chain = MicroServiceChain(
    microservices=deps,
    entry_point=deps[0],
    max_replicas = 5,
)

chain\
    .add_chain('d1', 'd2')\
    .add_chain('d2', 'd3')\
    .add_chain('d2', 'd4')\
    .add_chain('d4', 'd3')\
    .build()
# requests_paths = [RequestPath(deps[0], deps[1], deps[3], 5), RequestPath(deps[0], deps[1], deps[3], deps[2], 5)]
chain.add_request_path(
    RequestPath(
        microservices=[deps[0], deps[1], deps[2]],
        num_requests=1,
        load_types = [RequestLoadType.LOW_CPU_LOW_MEM, RequestLoadType.LOW_CPU_HIGH_MEM, RequestLoadType.HIGH_CPU_HIGH_MEM],
    ))
# ).add_request_path(
#     RequestPath(
#         microservices=[deps[0], deps[1], deps[3], deps[2]],
#         num_requests=5,
#         load_types = [RequestLoadType.LOW_CPU_LOW_MEM, RequestLoadType.HIGH_CPU_LOW_MEM, RequestLoadType.HIGH_CPU_HIGH_MEM, RequestLoadType.HIGH_CPU_HIGH_MEM],
#     )
# )


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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
        self.conv2 = GCNConv(16, 16)
        self.layer1 = nn.Linear(16, 64)
        self.layer2 = nn.Linear(64, 128)
        self.graph_fc = nn.Linear(128, 64)
        self.out_layers = nn.ModuleList([nn.Linear(64, n_actions) for _ in range(n_microservices)])

        self.edge_index = edge_index.to(device)
        self.batch = batch.to(device)

        self.device = device


    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, x):

        # print(f"Input shape: {x.shape}")

        x = x.to(torch.float32).to(self.device)

        # print(f"Before conv {x.shape}")
        x = self.conv1(x, self.edge_index)
        # print(f"After conv {x.shape}")
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, self.edge_index)


        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        # print(f"Before global mean {x.shape}")
        x = global_mean_pool(x, self.batch)
        # print(f"After global mean {x.shape}")
        x = self.graph_fc(x)
        x = F.relu(x)

        out = torch.stack([net(x) for net in self.out_layers], dim=1)
        return out
    



# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

state = chain.reset()
chain.start()
# Get number of actions from gym action space
n_actions = chain.max_replicas
# Get the number of state observations
# n_observations = len(state)
num_node_features = 4
n_microservices = len(chain.microservices)

policy_net = ChainGraphDQN(
    num_node_features,
    n_actions,
    n_microservices,
    chain.edge_index,
    chain.batch,
    device=device
).to(device)

target_net = ChainGraphDQN(
    num_node_features,
    n_actions,
    n_microservices,
    chain.edge_index,
    chain.batch,
    device=device
).to(device)

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
criterion = nn.SmoothL1Loss()
memory = ReplayMemory(64)


steps_done = 0

def select_action(env, state):
    # print(state)
    global steps_done
    sample = random.random()
    # exponential decaying eps
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            out = policy_net(state)
            return out.argmax(2)
    else:
        out = torch.tensor([env.action_space.sample()], device=device)
        return out

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

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

    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print(f"From replay {state_batch.shape}, {len(batch.state)} {len(batch.state[0])}")
    r = policy_net(state_batch).squeeze()
    # print(f"r shape: {r.shape}, action_batch shape: {action_batch.shape}")
    state_action_values = r.gather(2, action_batch)

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
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = criterion(state_action_values, expected_state_action_values)
    # print(loss.item())
    # # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.detach().item()


if torch.cuda.is_available():
    num_episodes = 100
else:
    num_episodes = 100

losses = []
rewards = []

logging.info("Starting agent training")
for i_episode in range(num_episodes):
    episode_losses = []
    episode_rewards = []
    # Initialize the environment and get its state
    state = chain.reset()
    # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # for t in count():
    for t in range(32):
        action = select_action(chain, state)
        # print(action)
        next_state, reward, terminated, _ = chain.step(action.squeeze().cpu().numpy())
        reward = torch.tensor([reward], device=device)

        if terminated:
            next_state = None
        # else:
        #     next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        # print(f'Pushing: state with shape: {state.shape} and next_state with shape {next_state.shape}')
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        loss = optimize_model()
        if loss:
            episode_losses.append(loss)
            episode_rewards.append(reward.to('cpu'))

        if terminated:
            # episode_durations.append(t + 1)
            # plot_durations()
            break

    # Soft update of the target network's weights
    # theta' <- tau * theta + (1 − tau) * theta'
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)
    # print(episode_losses)
    avg_loss = np.mean(episode_losses)
    avg_reward = np.mean(episode_rewards)
    logging.debug(f"Episode {i_episode}: Avg loss: {avg_loss}, Avg reward: {avg_reward}")
    print(f"Episode {i_episode}: Avg loss: {avg_loss}, Avg reward: {avg_reward}")
    losses.append(avg_loss)
    rewards.append(avg_reward)

    if (i_episode + 1) % 10 == 0:
        torch.save(policy_net.state_dict(), f"models/policy_net_{i_episode + 1}.pth")
        torch.save(target_net.state_dict(), f"models/target_net_{i_episode + 1}.pth")




print('Agent has completed')