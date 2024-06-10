from types_ import DeploymentLoadBalancingStrategy, ChainableRequest, RequestStatus, PodPhase
from pod import Pod
from apscheduler.schedulers.background import BackgroundScheduler
import uuid
import random
import numpy as np

class Deployment:
    def __init__(
            self,
            name: str,
            replicas: int,
            pod_cpu_limit,
            pod_memory_limit,
            pod_arrival_rate,
            pod_service_rate,
            create=False,
            lb_strategy: DeploymentLoadBalancingStrategy = DeploymentLoadBalancingStrategy.LEAST_USED,
        ):
        self.deploy_id = 'deploy-' + uuid.uuid4().hex
        self.name = name
        self.replicas = replicas
        self.pod_cpu_limit = pod_cpu_limit
        self.pod_memory_limit = pod_memory_limit
        self.pod_arrival_rate = pod_arrival_rate
        self.pod_service_rate = pod_service_rate
        self.scheduler = BackgroundScheduler()
        self.pods = [Pod(
            name=f"pod-{r}-{name}",
            scheduler=self.scheduler,
            cpu_limit=pod_cpu_limit,
            memory_limit=pod_memory_limit,
            arrival_rate=pod_arrival_rate,
            service_rate=pod_service_rate,
            create=create,
        ) for r in range(replicas)]

        self.lb_strategy = lb_strategy

        # Start the deployment scheduler to execute pods jobs
        self.scheduler.start()

    def __call__(self, request: ChainableRequest, wait = False):
        return self.receive_request(request, wait = wait)

    def receive_request(self, request: ChainableRequest, wait):
        if self.is_ready():
            # Load balance requests between running pods randomly
            selected_pod = None
            if len(self.pods) == 0:
                # print(f"Deployment pods is empty {self.name}")
                request.status = RequestStatus.ABORTED
                return False

            if self.lb_strategy == DeploymentLoadBalancingStrategy.RANDOM:
                selected_pod = random.choice(self.pods)
            elif self.lb_strategy == DeploymentLoadBalancingStrategy.LEAST_USED:
                selected_pod = min(self.pods, key=lambda pod: pod.cpu_usage + pod.memory_usage)
            if not selected_pod:
                print("No pod selected")
                return False
            return selected_pod.receive_request(request, wait = wait)
        return False

    def scale(self, new_replicas, create = True):
        if (self.replicas > new_replicas) and len(self.pods) != 0:
            # Make scaling down action
            # First we need to designate pods to terminate
            # print("Scaling down")
            if new_replicas == 0:
                print(f"Warning: scaling deployment {self.name} to 0")

            # TODO: Wait for the queue to be empty first
            pods_to_terminate = random.choices(self.pods, k = self.replicas - new_replicas)
            for pod in pods_to_terminate:
                pod.stop()
            self.replicas = new_replicas
            return
        elif self.replicas < new_replicas:
            # Make scale up action
            # print("Scaling up")
            self.pods += [Pod(
                            name=f"pod-{r+len(self.pods)}-{self.name}",
                            scheduler=self.scheduler,
                            cpu_limit=self.pod_cpu_limit,
                            memory_limit=self.pod_memory_limit,
                            arrival_rate=self.pod_arrival_rate,
                            service_rate=self.pod_service_rate,
                            create=create,
                        ) for r in range(new_replicas - self.replicas)]

            self.replicas = new_replicas

            return

    def terminate_pod(self, pod: Pod):
        # self.pods = list(filter(lambda p: p.pod_id == pod.pod_id, self.pods))
        pod.stop()

    def stop(self):
        # Stop all pods first
        for pod in self.pods:
            pod.stop()
        # Stop the scheduler
        self.scheduler.shutdown()

    def get_latency(self):
        l = list(map(lambda pod: pod.get_pod_latency(), self.pods))
        return 0 if len(l) == 0 else np.mean(l)

    def is_ready(self):
        return all(map(lambda pod: pod.phase == PodPhase.RUNNING, self.pods))
    @property
    def cpu_usage(self):
        l = list(map(lambda pod: pod.cpu_usage, self.pods))
        # print(f"Pod: {self.name} CPU usage: {l}")
        return 0 if len(l) == 0 else np.mean(l)

    @property
    def memory_usage(self):
        l = list(map(lambda pod: pod.memory_usage, self.pods))
        # print(f"Pod: {self.name} Mem usage: {l}")
        return 0 if len(l) == 0 else np.mean(l)


    def __repr__(self):
        pod_reprs = ', '.join(repr(pod) for pod in self.pods)
        return (f"Deployment(name={self.name}, replicas={self.replicas}, pod_cpu_limit={self.pod_cpu_limit}, "
                f"pod_memory_limit={self.pod_memory_limit}, pod_arrival_rate={self.pod_arrival_rate}, "
                f"pod_service_rate={self.pod_service_rate}, pods=[{pod_reprs}])")
