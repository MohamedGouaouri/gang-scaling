import gym
import networkx as nx
import numpy as np
import threading
import torch
import matplotlib.pyplot as plt
import time
from typing import List
from deployment import Deployment
from load_generator import RequestPath, LoadGenerator
from types_ import ChainableRequest, RequestStatus
from torch_geometric.utils.convert import from_networkx
from gym import spaces

class MicroServiceChain(gym.Env):
    def __init__(
            self,
            microservices: List[Deployment],
            entry_point: Deployment,
            max_replicas: int = 20,
            slo_reward_factor = 0.7,
        ):
        self.microservices = microservices
        self.entry_point = entry_point

        self._build_graph_map()
        # self.graph = dict()
        self.graph = nx.DiGraph()
        self.max_replicas = max_replicas

        # Define action space: number of replicas for each microservice (Discrete space)
        self.action_space = spaces.MultiDiscrete([max_replicas] * len(self.microservices))  # Assume min 1 and max replicas for each microservice

        # Define observation space: latencies, CPU usage, memory usage, and number of replicas for each deployment
        # This is a type of space in Gym that represents a continuous n-dimensional box.
        # It is used to specify that the observations are vectors of real numbers with specified ranges for each dimension.
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(len(self.microservices) * 4,),
            dtype=np.float32
        )

        self.slo_reward_factor = slo_reward_factor
        self.resource_reward_factor = 1 - self.slo_reward_factor

        self.requests_paths = []
        self._built = False
        self.edge_index = None
        self.batch = None

        self.n_aborted_requests  = 0
        self.cool_down_period = 5

        self.scheduler = threading.Thread(target=self.generate_load_continuously, daemon=True)


    def add_chain(self, source: str, target: str):
        # First, we need to make sure that source and target are part of the graph
        assert source in self.graph_map and target in self.graph_map
        self.graph.add_edge(source, target)
        return self

    def add_request_path(self, request_path: RequestPath):
        self.requests_paths.append(request_path)
        return self

    def build(self):
        self.load_generator = LoadGenerator(requests_paths=self.requests_paths,)
        self._update_graph_attr()
        self._built = True
        # Run the generator
        # self.scheduler.add_job(self._generate_load, 'interval', seconds=1/10, max_instances=1000)
        # self.scheduler.start()


    def _build_graph_map(self):
        self.graph_map = {microservice.name: microservice for microservice in self.microservices}

    def __call__(self, request: ChainableRequest):
        # A method to request to the chain of microservices
        self.entry_point(request, wait = True)


    def step(self, action):
        # Apply the action (scale the deployments)
        for i, replicas in enumerate(action):
            # print(replicas)
            self.microservices[i].scale(replicas + 1)

        time.sleep(self.cool_down_period)

        # Get the new state
        state = self._get_graph_state()

        # Calculate reward
        reward = self._calculate_reward(state)

        # Check if the episode is done
        done = self._check_done(state)


        self._update_graph_attr()

        return state, reward, done, {}

    def _get_state(self):
        # Collect the state information from the deployments
        state = []
        for deployment in self.microservices:
            latency = deployment.get_latency()
            cpu_usage = deployment.cpu_usage
            memory_usage = deployment.memory_usage
            replicas = deployment.replicas
            state.extend([latency, cpu_usage, memory_usage, replicas])

        return np.array(state, dtype=np.float32)

    def _get_graph_state(self):

        data = from_networkx(self.graph)
        data.x = np.array([data.replicas, data.latency, data.cpu_usage, data.memory_usage]).T
        data.x = torch.from_numpy(data.x)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        del data.replicas
        del data.latency
        del data.cpu_usage
        del data.memory_usage

        self.edge_index = data.edge_index
        self.batch = data.batch

        return data.x

    def _update_graph_attr(self):
        feature_map = {}
        for deployment in self.microservices:
            latency = deployment.get_latency()
            cpu_usage = deployment.cpu_usage
            memory_usage = deployment.memory_usage
            replicas = deployment.replicas
            feature_map[deployment.name] = {
                'replicas': replicas,
                'latency': latency,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage
            }
        nx.set_node_attributes(self.graph, feature_map)

    def _calculate_reward(self, state):
        # Calculate the reward based on the state
        # Example: minimize the latencies while penalizing high resource usage
        # replicas = state[0::4]
        latencies = state[:, 0]
        cpu_usages = state[:, 1]
        memory_usages = state[:, 2]
        # print(state)

        reward = - self.slo_reward_factor * torch.mean(latencies)  # Minimize total latency
        reward -= self.resource_reward_factor * torch.mean(cpu_usages)  # Penalize high CPU usage
        reward -= self.resource_reward_factor * torch.mean(memory_usages)  # Penalize high memory usage
        # reward -= np.mean(replicas)  # Penalize using more replicas
        reward -= self.n_aborted_requests * 100

        return reward

    def reset(self):
        # Reset the environment to an initial state
        for deployment in self.microservices:
            deployment.scale(1)  # Reset to 1 replica per deployment

        self._update_graph_attr()

        return self._get_graph_state()

    def _check_done(self, state):
        # Check if the episode is done
        # Example: stop if latency exceeds a threshold
        latencies = state[:, 0]
        cpu_usages = state[:, 1]
        memory_usages = state[:, 2]
        # Arbitrary threshold
        if torch.any(latencies > 10) or torch.any(cpu_usages >= 1.0) or torch.any(memory_usages >= 1.0):
            return True
        return False


    def generate_load_continuously(self):
        while True:
            # Generate load here
            self._generate_load()
            time.sleep(1)  # Adjust the sleep time as necessary

    def _generate_load(self,):
        generated_requests = self.load_generator.generate()
        for rp in generated_requests:
            for req in rp:
                self.__call__(req)
                if req.status == RequestStatus.ABORTED:
                    self.n_aborted_requests += 1


    def start(self):
        self.scheduler.start()



    def draw(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', arrowsize=20)
        plt.show()

    def stop(self):
        for deployment in self.microservices:
            deployment.stop()
        # self.scheduler.shutdown()
