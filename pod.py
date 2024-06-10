from collections import deque
from types_ import PodPhase, ChainableRequest, RequestStatus, PodOverloadedException

import queue
import uuid
import numpy as np
import matplotlib.pyplot as plt
import time
from contextlib import contextmanager


class Pod:
    def __init__(self, name, scheduler, cpu_limit, memory_limit, arrival_rate, service_rate, create=False):
        self.pod_id = 'pod-' + uuid.uuid4().hex
        self.name = name
        self.phase = PodPhase.PENDING
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.cpu_usage = 0
        self.memory_usage = 0
        self.request_queue = queue.Queue()
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate

        self.last_latencies = deque(maxlen=100)
        self._is_stopped = False

        self.scheduler = scheduler
        self.job = None


        if create:
            self.create()


    @contextmanager
    def scheduler_context(self):
        # Schedul the new job
        try:
            yield ""
        finally:
            self.scheduler.add_job(self.run)

    def create(self):
        self.phase = PodPhase.PENDING
        self.job = self.scheduler.add_job(self.run)
        self.phase = PodPhase.RUNNING

    def run(self):
        if self.phase == PodPhase.RUNNING and not self._is_stopped:
            self.process_requests()

    def receive_request(self, request: ChainableRequest, wait: bool = False):
        with self.scheduler_context():
            if self.phase == PodPhase.RUNNING:
              # if self.request_queue.empty()
              request.status = RequestStatus.QUEUED
              self.request_queue.put(request)
              if wait:
                  while request.status != RequestStatus.RESOLVED and request.status != RequestStatus.ABORTED:
                    #   print(f"Waiting for request {request.request_id} to be completed")
                      time.sleep(0.001)
                      # continue
                  if request.status == RequestStatus.RESOLVED:
                      return True
                  return False
              return True
            return False

    def process_requests(self):
        if not self.request_queue.empty():
            request = self.request_queue.get_nowait()
            try:
                self.execute(request)
            except PodOverloadedException as e:
                request.retries += 1
                if request.retries > request.max_retries:
                    request.status = RequestStatus.ABORTED
                else:
                    request.status = RequestStatus.QUEUED
                    self.request_queue.put(request)
            finally:
                request.status = RequestStatus.RESOLVED
                self.request_queue.task_done()
                # print(f"Request {request.request_id} done")


    def execute(self, request: ChainableRequest):

        # Calculate cpu and memory consumption
        ## cpu unit is millicpu used by k8s
        ## memory unit is megabytes used by k8s
        prev_cpu_usage = self.cpu_usage
        prev_memory_usage = self.memory_usage

        future_cpu_usage = self.cpu_limit * self.cpu_usage
        future_memory_usage = self.memory_limit * self.memory_usage

        future_cpu_usage += request.cpu_consumption
        future_memory_usage += request.memory_consumption

        # print("Future usages: ", self.cpu_limit, future_cpu_usage, self.memory_limit, future_memory_usage)
        if future_cpu_usage > self.cpu_limit or future_memory_usage > self.memory_limit:
            raise PodOverloadedException("Pod is overloaded")  # System is overloaded

        self.cpu_usage = future_cpu_usage / self.cpu_limit
        self.memory_usage = future_memory_usage / self.memory_limit

        # Calculate latency
        processing_latency = self.get_processing_latency()
        queuing_latency = self.get_queuing_latency()


        # Update request's local latencies
        request.processing_latency = processing_latency
        request.queuing_latency = queuing_latency

        # Unbox request and call next requests
        if request.next_request:
            # Next request processing times affect this pod
            request.next_request.to(request.next_request.request, wait = True) # Don't we need to wait ?
            processing_latency += request.next_request.request.processing_latency
            # queuing_latency += request.next_request.request.queuing_latency

        # Append subsequent latencies to the last latencies buffer
        self.last_latencies.append(queuing_latency + processing_latency)

        ## Reset cpu and mem usage after processing
        self.cpu_usage = prev_cpu_usage
        self.memory_usage = prev_memory_usage
        # print(f"Pod: {self.name} {queuing_latency + processing_latency}")


    def get_processing_latency(self):
        # Calculate processing latency
        ## Calculate adjusted service rate based on CPU and memory usage
        adjusted_service_rate = self.service_rate * (1 - self.cpu_usage) * (1 - self.memory_usage)
        ## Number of requests in the queue
        # queue_size = self.request_queue.qsize()
        ## Calculate response time using M/M/1 formula
        processing_latency = 1 / (adjusted_service_rate + 0.000001)
        # print(f"Processing: {processing_latency}")
        return processing_latency

    def get_queuing_latency(self):
        adjusted_service_rate = self.service_rate * (1 - self.cpu_usage) * (1 - self.memory_usage)
        queuing_latency = self.arrival_rate / ( adjusted_service_rate  * abs( adjusted_service_rate - self.arrival_rate ) + 0.000001)
        # print(f"Queueing: {queuing_latency}")
        return queuing_latency

    def plot_latency_diagram(self):
        x = np.linspace(0, 10, 10)
        y = list(self.last_latencies) + [0 for _ in range(10 - len(self.last_latencies)) ]
        plt.plot(x, y)

    def stop(self):
        self.phase = PodPhase.SUCCEEDED
        self._is_stopped = True
        if self.job:
            try:
                self.job.remove()
            except:
                pass

    def get_pod_latency(self):
        l = list(self.last_latencies)
        return 0 if len(l) == 0 else np.mean(l)

    def update_configs(self):
        # TODO: Implement this
        pass
    def status(self):
        # TODO: Implement this
        return {
            "name": self.name,
            "phase": self.phase,
        }
    def __repr__(self):
        return (f"Pod(name={self.name}, phase={self.phase}, cpu_usage={self.cpu_usage}, "
                f"memory_usage={self.memory_usage}, cpu_limit={self.cpu_limit}, "
                f"memory_limit={self.memory_limit}, arrival_rate={self.arrival_rate}, "
                f"service_rate={self.service_rate})")
