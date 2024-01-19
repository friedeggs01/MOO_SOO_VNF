from .node import *

class Link():
    def __init__(self, source:Node, destination:Node, delay: float, resource_capacity: float = -1, resource_available: float = -1) -> None:
        self.source = source
        self.destination = destination
        self.id = "{}-{}".format(self.source.id, self.destination.id)
        self.delay = delay
        self.total_delay = 0
        self.resource_capacity = resource_capacity
        self.resource_available = resource_available

    def consume(self, require_value):
        if require_value > self.resource_available:
            print(f"Link {self.id} does not have enough bandwidth")
            return False
        else:
            self.resource_available -= require_value
            self.total_delay += self.delay
            return True

    def ratio_performance(self) -> float:
        return self.resource_available / self.resource_capacity
        
    def __repr__(self) -> str:
        return "Link: {} -> {} | Delay: {} | Bandwidth: {}/{}".format(self.source.id, self.destination.id, self.delay, self.resource_available, self.resource_capacity)