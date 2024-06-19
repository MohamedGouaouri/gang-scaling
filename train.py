from deployment import Deployment
from envs import MicroServiceChain, HPAEnv
from load_generator import RequestPath
from types_ import RequestLoadType
from agents import ChainGraphDQN, ReplayMemory, select_action, optimize_model, device, BATCH_SIZE, LR, TAU
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os
import warnings
from utils import save_performance_plots, exponential_moving_average
from datetime import datetime

# Create a directory to save models
os.makedirs("models", exist_ok=True)
experiment_name = 'Giyu_Tomioka_' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Set up logging
logging.basicConfig(filename='training.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

logging.getLogger('apscheduler.executors.default').setLevel(logging.ERROR)
logging.getLogger('apscheduler.executors.default').propagate = False

deps1 = [
    Deployment(
        name="frontend",
        replicas=1,
        pod_cpu_limit=4000,
        pod_memory_limit=2**30, # 1 gb
        pod_arrival_rate=10,
        pod_service_rate=20,
        create=True,
        # lb_strategy=DeploymentLoadBalancingStrategy.RANDOM
    ),
    Deployment(
        name="backend",
        replicas=2,
        pod_cpu_limit=4000, # 4 cpus
        pod_memory_limit=2**30, # 1 gb
        pod_arrival_rate=5,
        pod_service_rate=4,
        create=True,
        # lb_strategy=DeploymentLoadBalancingStrategy.RANDOM
    ),
    Deployment(
        name="database",
        replicas=3,
        pod_cpu_limit=2000, # 1 cpus
        pod_memory_limit=0.5 * 2**30, # 1 gb
        pod_arrival_rate=10,
        pod_service_rate=12,
        create=True,
        # lb_strategy=DeploymentLoadBalancingStrategy.RANDOM
    ),
    Deployment(
        name="redis",
        replicas=2,
        pod_cpu_limit=2000, # 3 cpus
        pod_memory_limit=0.5 * 2**30, # 1 gb
        pod_arrival_rate=5,
        pod_service_rate=10,
        create=True,
        # lb_strategy=DeploymentLoadBalancingStrategy.RANDOM

    )
]

deps2 = [
    Deployment(
        name="frontend",
        replicas=1,
        pod_cpu_limit=4000,
        pod_memory_limit=2**30, # 1 gb
        pod_arrival_rate=10,
        pod_service_rate=20,
        create=True,
        # lb_strategy=DeploymentLoadBalancingStrategy.RANDOM
    ),
    Deployment(
        name="backend",
        replicas=2,
        pod_cpu_limit=4000, # 4 cpus
        pod_memory_limit=2**30, # 1 gb
        pod_arrival_rate=5,
        pod_service_rate=4,
        create=True,
        # lb_strategy=DeploymentLoadBalancingStrategy.RANDOM
    ),
    Deployment(
        name="database",
        replicas=3,
        pod_cpu_limit=2000, # 1 cpus
        pod_memory_limit=0.5 * 2**30, # 1 gb
        pod_arrival_rate=10,
        pod_service_rate=12,
        create=True,
        # lb_strategy=DeploymentLoadBalancingStrategy.RANDOM
    ),
    Deployment(
        name="redis",
        replicas=2,
        pod_cpu_limit=2000, # 3 cpus
        pod_memory_limit=0.5 * 2**30, # 1 gb
        pod_arrival_rate=5,
        pod_service_rate=10,
        create=True,
        # lb_strategy=DeploymentLoadBalancingStrategy.RANDOM

    )
]

chain = MicroServiceChain(
    microservices=deps1,
    entry_point=deps1[0],
    max_replicas = 10,
)

hpa_chain = HPAEnv(
    microservices=deps2,
    entry_point=deps2[0],
    max_replicas = 10,
)


chain\
    .add_chain('frontend', 'backend')\
    .add_chain('backend', 'database')\
    .add_chain('backend', 'redis')\
    .add_chain('redis', 'database')\
    .build()

hpa_chain\
    .add_chain('frontend', 'backend')\
    .add_chain('backend', 'database')\
    .add_chain('backend', 'redis')\
    .add_chain('redis', 'database')\
    .build()


# requests_paths = [RequestPath(deps[0], deps[1], deps[3], 5), RequestPath(deps[0], deps[1], deps[3], deps[2], 5)]

rp1 = RequestPath(
        microservices=[deps1[0], deps1[1], deps1[2]],
        num_requests=20,
        load_types = [RequestLoadType.LOW_CPU_LOW_MEM, RequestLoadType.LOW_CPU_HIGH_MEM, RequestLoadType.HIGH_CPU_HIGH_MEM],
  )

rp2 = RequestPath(
        microservices=[deps1[0], deps1[1], deps1[3], deps1[2]],
        num_requests=20,
        load_types = [RequestLoadType.LOW_CPU_LOW_MEM, RequestLoadType.HIGH_CPU_LOW_MEM, RequestLoadType.HIGH_CPU_HIGH_MEM, RequestLoadType.HIGH_CPU_HIGH_MEM],
    )
chain.add_request_path(rp1).add_request_path(rp2)





# requests_paths = [RequestPath(deps[0], deps[1], deps[3], 5), RequestPath(deps[0], deps[1], deps[3], deps[2], 5)]

rp1 = RequestPath(
        microservices=[deps2[0], deps2[1], deps2[2]],
        num_requests=20,
        load_types = [RequestLoadType.LOW_CPU_LOW_MEM, RequestLoadType.LOW_CPU_HIGH_MEM, RequestLoadType.HIGH_CPU_HIGH_MEM],
  )

rp2 = RequestPath(
        microservices=[deps2[0], deps2[1], deps2[3], deps2[2]],
        num_requests=20,
        load_types = [RequestLoadType.LOW_CPU_LOW_MEM, RequestLoadType.HIGH_CPU_LOW_MEM, RequestLoadType.HIGH_CPU_HIGH_MEM, RequestLoadType.HIGH_CPU_HIGH_MEM],
    )
chain.add_request_path(rp1).add_request_path(rp2)

hpa_chain.add_request_path(rp1).add_request_path(rp2)







hpa_chain.reset()
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
memory = ReplayMemory(16)

if torch.cuda.is_available():
    num_episodes = 200
else:
    num_episodes = 100


n_runs = 10

if torch.cuda.is_available():
    num_episodes = 1000
else:
    num_episodes = 1


print("Agent training started")
logging.info(f"Agent training started")

losses = []
rewards = []
hpa_rewards = []
for i_episode in range(num_episodes):
    episode_losses = []
    episode_rewards = []
    hpa_episode_rewards = []

    # Initialize the environment and get its state
    state = chain.reset()
    # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # for t in count():
    for t in range(16):
        # print(f"State in loop {state}")
        action = select_action(chain, policy_net, state).squeeze()
        # print(action)
        next_state, reward, terminated, _ = chain.step(action.cpu().numpy())
        _, hpa_reward, _, _ = hpa_chain.step()

        reward = torch.tensor([reward], device=device)

        if terminated:
            next_state = None

        # Store the transition in memory
        # print(f'Pushing: state with shape: {state.shape} and next_state with shape {next_state.shape}')
        memory.push(state, action, next_state, reward)

        if terminated:
            # episode_durations.append(t + 1)
            # plot_durations()
            break

        state = next_state

        if len(memory) < BATCH_SIZE:
            continue
        # Move to the next state

        # Perform one step of the optimization (on the policy network)
        loss = optimize_model(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            criterion=criterion,
            memory=memory
        )
        if loss:
            episode_losses.append(loss)
            episode_rewards.append(reward.to('cpu'))
            hpa_episode_rewards.append(hpa_reward)


    # Soft update of the target network's weights
    # theta' <- tau * theta + (1 − tau) * theta'
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)

    logging.debug(f"Avg episode loss: {np.mean(episode_losses)}, Avg episode reward: {np.mean(episode_rewards)}")
    logging.debug(f"HPA: Avg episode reward: {np.mean(hpa_episode_rewards)}")


    losses.append(np.mean(episode_losses))
    rewards.append(np.mean(episode_rewards))
    hpa_rewards.append(np.mean(hpa_episode_rewards))

    if (i_episode + 1) % 10 == 0:
        save_performance_plots(losses, rewards, experiment_name + f" episode_{i_episode}")
        save_performance_plots(
            exponential_moving_average(losses, 5), 
            exponential_moving_average(rewards, 5), 
            experiment_name + f" episode_{i_episode}"
        )

torch.save(policy_net.state_dict(), f"models/policy_net_{experiment_name}.pth")
torch.save(target_net.state_dict(), f"models/target_net_{experiment_name}.pth")
save_performance_plots(losses, rewards, experiment_name)
save_performance_plots(exponential_moving_average(losses, 5), exponential_moving_average(rewards, 5), experiment_name)

