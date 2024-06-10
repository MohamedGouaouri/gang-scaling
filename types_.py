from enum import Enum
from typing import Optional
import random

class PodPhase(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"

    def __repr__(self):
        return f"<PodPhase: {self.value}>"

class RequestStatus(Enum):
    NOT_STARTED = "NotStarted"
    QUEUED = "Queued"
    PROCESSING = "Processing"
    RESOLVED = "Resolved"
    ABORTED = "Aborted"


class RequestLoadType(Enum):
    HIGH_CPU_HIGH_MEM = "HighCpu_HighMem"
    HIGH_CPU_LOW_MEM = "HighCpu_LowMem"
    LOW_CPU_HIGH_MEM = "LowCpu_HighMem"
    LOW_CPU_LOW_MEM = "LowCpu_LowhMem"

    def get_metrics(self):
        mapping = {
            RequestLoadType.HIGH_CPU_HIGH_MEM: (1000, 100 * 2 ** 20),   # 1 CPU core and 100 MB
            RequestLoadType.HIGH_CPU_LOW_MEM: (100, 1 * 2 ** 20),       # 1 CPU core and 1 MB
            RequestLoadType.LOW_CPU_HIGH_MEM: (1, 100 * 2 ** 20),       # 10% CPU core and 100 MB
            RequestLoadType.LOW_CPU_LOW_MEM: (1, 1 * 2 ** 20),          # 10% CPU core and 1 MB
        }
        base = mapping[self]

        # Add Gaussian noise
        cpu_noise = random.gauss(0, 0.1 * base[0])  # mean=0, std_dev=10% of the base CPU value
        mem_noise = random.gauss(0, 0.1 * base[1])  # mean=0, std_dev=10% of the base memory value

        cpu = base[0] + cpu_noise
        mem = base[1] + mem_noise
        return (cpu, mem)


class DeploymentLoadBalancingStrategy(Enum):
    RANDOM = "Random"
    LEAST_USED = "LeastUsed"
    ROUND_ROBIN = "RoundRobin"
    # TODO: Add other strategies
    def __repr__(self):
        return f"<DeploymentLoadBalancingStrategy: {self.value}>"

class PodRequest:
    def __init__(self,
                 request_id: int,
                 cpu_consumption: float,
                 memory_consumption: float,
                 queuing_latency = 0.0,
                 processing_latency = 0.0,
                 status = RequestStatus.NOT_STARTED,
                 load_type = RequestLoadType.LOW_CPU_LOW_MEM,):
        self.request_id = request_id
        self.cpu_consumption = cpu_consumption
        self.memory_consumption = memory_consumption
        self.queuing_latency = queuing_latency
        self.processing_latency = processing_latency
        self.status = status
        self.load_type = load_type

    def __repr__(self):
        return (f"<PodRequest(id={self.request_id}, cpu={self.cpu_consumption}, "
                f"memory={self.memory_consumption}, queuing_latency={self.queuing_latency}, "
                f"processing_latency={self.processing_latency})>")

class ChainableRequest(PodRequest):
    def __init__(self,
                 request_id: int,
                 cpu_consumption: float,
                 memory_consumption: float,
                 queuing_latency = 0,
                 processing_latency = 0,
                 next_request: Optional['NextRequest'] = None,
                 status = RequestStatus.NOT_STARTED,
                 load_type = RequestLoadType.LOW_CPU_LOW_MEM,

        ):
        super().__init__(request_id, cpu_consumption, memory_consumption, queuing_latency, processing_latency,status, load_type)
        self.next_request = next_request
        self.retries = 0
        self.max_retries = 0


    def total_processing_latency(self):
        current = self
        total = current.processing_latency
        while current.next_request:
            current = current.next_request.request
            total += current.processing_latency
        return total
    def total_queuing_latency(self):
        current = self
        total = current.queuing_latency
        while current.next_request:
            current = current.next_request.request
            total += current.queuing_latency
        return total

    def total_latency(self):
        return self.total_processing_latency() + self.total_queuing_latency()

    def __repr__(self):
        return (f"<ChainableRequest(id={self.request_id}, cpu={self.cpu_consumption}, "
                f"memory={self.memory_consumption}, queuing_latency={self.queuing_latency}, "
                f"processing_latency={self.processing_latency}, retries={self.retries}, "
                f"max_retries={self.max_retries}, next_request={repr(self.next_request)})>")

    def visualize(self, indent=0):
        indent_str = '  ' * indent
        repr_str = (f"{indent_str}ChainableRequest(id={self.request_id}, cpu={self.cpu_consumption}, "
                    f"memory={self.memory_consumption}, queuing_latency={self.queuing_latency}, "
                    f"processing_latency={self.processing_latency}, retries={self.retries}/{self.max_retries})")
        if self.next_request:
            print(f"{indent_str}  NextRequest to {self.next_request.to.name}:")
            self.next_request.request.visualize(indent + 2)
class NextRequest:
    def __init__(self, request: ChainableRequest, to):
        self.request = request
        self.to = to
    def __repr__(self):
        return f"<NextRequest(to={self.to}, request={repr(self.request)})>"



class PodOverloadedException(Exception):
    def __init__(self, message: str = 'Pod is overloaded'):
        super().__init__(message)

class RequestMaxRetriesReached(Exception):
    def __init__(self, message: str = 'RequestMaxRetriesReached'):
        super().__init__(message)