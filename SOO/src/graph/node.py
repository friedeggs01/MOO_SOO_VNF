from .vnf import *

# for new formulation
class Node():
    def __init__(self, id: int, type: bool, delay: float, cost: float,
                mem_capacity: float = -1, mem_available: float = -1,
                cpu_capacity: float = -1, cpu_available: float = -1,
                vnf_used: list = None, vnf_possible: list = None, vnf_cost: list = None, num_vnfs_limit: int = -1
                ) -> None:
        self.id = id
        self.delay = delay
        self.cost = cost
        self.mem_capacity = mem_capacity
        self.mem_available = mem_available
        self.cpu_capacity = cpu_capacity
        self.cpu_available = cpu_available
        self.type = type
        self.type_str = "Server" if self.type else "SNNode"
        self.vnf_used = vnf_used
        self.vnf_possible = vnf_possible
        self.vnf_cost = vnf_cost
        self.num_vnfs_limit = num_vnfs_limit
        self.total_delay = 0
        if self.vnf_cost:
            self.total_vnf_cost = sum(self.vnf_cost)
        self.total_installed_vnf_cost = 0

    def consume_mem(self, require_value) -> bool:
        if self.type == 0:
            if require_value > self.mem_available:
                return False
            else:
                self.mem_available -= require_value
                return True
        else: # server doesn't consume memory
            return True
    
    def consume_cpu(self, require_value) -> bool:
        if self.type == 0: raise Exception
        if require_value > self.cpu_available:
            return False
        else:
            self.cpu_available -= require_value
            self.total_delay += self.delay
            return True

    def install_vnf(self, type) -> bool:
        if type in self.vnf_possible and self.check_num_vnf(type):
            if type not in self.vnf_used:
                self.total_installed_vnf_cost += self.vnf_cost[type]
            self.vnf_used.append(type)
            return True
        else:
            return False

    def check_num_vnf(self, vnf_id):
        vnf_lst = self.vnf_used + [vnf_id]
        if len(set(vnf_lst)) <= self.num_vnfs_limit:
            return True
        else:
            return False

    def ratio_performance(self) -> float:
        return self.mem_available / self.mem_capacity

    def __repr__(self, ) -> str:
        if self.type == 1:
            return "{} | ID: {} | Delay: {} | Cost: {} | CPU: {}/{} | vnf_used: {} | total_vnf_installing: {}".format(self.type_str, self.id, self.delay, self.cost, 
                                                                        self.cpu_available, self.cpu_capacity,                                                                         self.vnf_used, self.total_installed_vnf_cost)
        else:
            return "{} | ID: {} | Delay: {} | Cost: {} | Mem: {}/{}".format(self.type_str, self.id, self.delay, self.cost, 
                                                                        self.mem_available, self.mem_capacity)
