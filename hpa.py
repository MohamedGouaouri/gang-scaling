from deployment import Deployment
from apscheduler.schedulers.background import BackgroundScheduler
import time

class HPA:
  def __init__(self, deployment: Deployment, target_cpu_usage, min_replicas, max_replicas):
    self.deployment = deployment
    self.target_cpu_usage = target_cpu_usage
    self.min_replicas = min_replicas
    self.max_replicas = max_replicas
    self.scheduler = BackgroundScheduler()
    self.cool_down_period = 3


  def start(self):
    self.scheduler.add_job(self.run)
    self.scheduler.start()
  def stop(self):
    self.scheduler.shutdown()

  def run(self):
    current_cpu_usage = self.deployment.cpu_usage
    current_replicas = self.deployment.replicas
    new_replicas = int(current_replicas * current_cpu_usage / self.target_cpu_usage)
    if (new_replicas < self.min_replicas):
      new_replicas = self.min_replicas
    elif (new_replicas > self.max_replicas):
      new_replicas = self.max_replicas

    self.deployment.scale(new_replicas)
    time.sleep(self.cool_down_period)
    self.scheduler.add_job(self.run)

