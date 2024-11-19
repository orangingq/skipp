from datetime import timedelta
import time
from typing import List

class Metric():
    '''Metric class to record the performance of the model.'''
    def __init__(self, name:str, default:float, step_size:int=1):
        self.name = name
        self.default = default
        self.value = default
        self.__step = 0
        self.step_size = step_size
    def reset(self):
        self.value = self.default
        self.__step = 0
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"
    def step(self, value:float):
        self.__step += self.step_size
        self.value += value
    def get_step(self):
        return self.__step
    def __eq__(self, other):
        return self.value == other
    def __ne__(self, other):
        return self.value != other
    def __lt__(self, other):
        return self.value < other
    def __le__(self, other):
        return self.value <= other
    def __gt__(self, other):
        return self.value > other
    def __ge__(self, other):
        return self.value >= other
    @property
    def avg(self) -> float:
        if self.__step == 0:
            return self.default
        return float(self.value) / self.__step
    
class TimeMetric(Metric):
    '''TimeMetric class to record the time taken for the model to train.'''
    def __init__(self, name:str, since:float):
        super().__init__(name, since)
        self.since = since
    def elapsed(self, now:float=None)->float:
        if now is None:
            now = time.time()
        return now - self.since
    def reset(self, since:float=None):
        if since is None:
            since = time.time()
        self.since = since
    def __str__(self) -> str:
        return f"{self.name}: {str(timedelta(seconds=self.elapsed()))}"
    def step(self, value:float):
        self.since = value

class MetricGroup():
    '''MetricGroup class to record the performance of the model'''
    def __init__(self, metrics:List[Metric]):
        self.metrics = metrics
    def reset(self, metrics:List[Metric]=None):
        if metrics is None:
            metrics = self.metrics
        for metric in metrics:
            metric.reset()
    def __str__(self) -> str:
        return " | ".join([str(metric) for metric in self.metrics])
    def step(self, values:List[float]):
        for metric, value in zip(self.metrics, values):
            metric.step(value)
    @property
    def avg(self) -> List[float]:
        return [metric.avg for metric in self.metrics]