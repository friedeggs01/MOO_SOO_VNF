from .node import *
    
class Link():
    def __init__(self, source: Node, destination: Node, 
                 delay: float, bw_capacity = float('inf'), 
                 bw_available = float('inf')) -> None:
        self.source = source
        self.destination = destination
        self.id = "{}-{}".format(self.source.id, self.destination.id)
        self.delay = delay
        self.total_delay = 0
        self.bw_capacity = bw_capacity
        self.bw_available = bw_available
        
    def bandwidth_consume(self, require_bw):
        if require_bw > self.bw_available:
            print(f"Link {self.id} does not have enough bandwidth")
            return False
        else:
            self.bw_available -= require_bw
            self.total_delay += self.delay
            return True

    def ratio_performance(self) -> float:
        return self.resource_available / self.resource_capacity