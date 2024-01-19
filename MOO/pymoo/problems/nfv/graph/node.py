class Node():
    def __init__(self, id: int, type: bool, delay: float, cost: float,
                 mem_capacity: float = -1, mem_available = float('inf'),
                 cpu_capacity: float = -1, cpu_available = float('inf'),
                 vnf_used: list = None, vnf_possible: list = None,
                 vnf_cost: list = None, num_vnfs_limit: int = -1
                 ) -> None:
        self.id = id
        self.type = type
        self.type_str = "server" if self.type else "switch"
        self.delay = delay
        self.cost = cost
        self.mem_capacity = mem_capacity
        self.mem_available = mem_available
        self.cpu_capacity = cpu_capacity
        self.cpu_available = cpu_available
        self.vnf_used = vnf_used
        self.vnf_possible = vnf_possible
        self.vnf_cost = vnf_cost
        self.num_vnfs_limit = num_vnfs_limit
        self.total_delay = 0
        if self.vnf_cost:
            self.total_vnf_cost = sum(self.vnf_cost)
        self.total_installed_vnf_cost = 0
    
    def mem_consume(self, require_mem) -> bool:
        # only switch node consume memory
        if self.type == 0: 
            if require_mem > self.mem_available:
                return False
            else:
                self.mem_available -= require_mem
                return True
        else: 
            return True
    
    def cpu_consume(self, require_cpu):
        # only server node consume cpu
        if self.type == 0:
            return True
        else:
            if require_cpu > self.cpu_available:
                return False
            else:
                self.cpu_available -= require_cpu
                self.total_delay += self.delay
                return True
    

    def install_vnf(self, type) -> bool:
        if type in self.vnf_possible and self._check_num_vnf(type):
            if type not in self.vnf_used:
                self.total_installed_vnf_cost += self.vnf_cost[type]
            self.vnf_used.append(type)
            return True
        else:
            return False

    def _check_num_vnf(self, vnf_id):
        vnf_lst = self.vnf_used + [vnf_id]
        if len(set(vnf_lst)) <= self.num_vnfs_limit:
            return True
        else:
            return False

    def ratio_performance(self) -> float:
        return self.mem_available / self.mem_capacity
