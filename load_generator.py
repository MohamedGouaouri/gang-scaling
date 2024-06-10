from deployment import Deployment
from typing import List
from types_ import RequestLoadType, ChainableRequest, NextRequest
import numpy as np

class RequestPath:
    # Request path represents the request path information
    def __init__(self, microservices: List[Deployment], num_requests: int, load_types: List[RequestLoadType]):
        self.microservices = microservices
        self.num_requests = num_requests
        self.load_types = load_types

        assert len(self.microservices) == len(self.load_types)



class LoadGenerator:
    def __init__(self, requests_paths,):
        self.requests_paths = requests_paths

    def generate(self):
        requests = []
        for rpi, rp in enumerate(self.requests_paths):
            microservices = rp.microservices
            cpu_loads, mem_loads = self.generate_for_path(rp)
            rp_requests = []
            for i in range(rp.num_requests):
                idx = 0
                while idx < len(microservices) - 1:
                    request = ChainableRequest(
                        request_id = (rpi + 1) * i + idx,
                        cpu_consumption=cpu_loads[idx][i],
                        memory_consumption=mem_loads[idx][i]
                    )
                    next_request = ChainableRequest(
                        request_id= (rpi + 1) * i + idx + 1,
                        cpu_consumption=cpu_loads[idx + 1][i],
                        memory_consumption=mem_loads[idx + 1][i]
                    )

                    request.next_request = NextRequest(
                        to=microservices[idx+1],
                        request=next_request
                    )

                    idx += 1
                rp_requests.append(request)
            requests.append(rp_requests)
        return requests


    def generate_for_path(self, request_path: RequestPath):
        cpu_loads = []
        mem_loads = []
        for i, ms in enumerate(request_path.microservices):
            load_type = request_path.load_types[i]
            ms_cpu_load = self._generate_cpu(request_path.num_requests, n_replicas = ms.replicas, load_type = load_type)
            ms_mem_load = self._generate_mem(request_path.num_requests, n_replicas = ms.replicas, load_type = load_type)

            cpu_loads.append(ms_cpu_load)
            mem_loads.append(ms_mem_load)

        return cpu_loads, mem_loads



    def _generate_cpu(self, num_requests: int, n_replicas: int, load_type: RequestLoadType):
        return np.array([load_type.get_metrics()[0] for _ in range(num_requests)])

    def _generate_mem(self, num_requests: int, n_replicas: int, load_type: RequestLoadType):
        return np.array([load_type.get_metrics()[1] for _ in range(num_requests)])