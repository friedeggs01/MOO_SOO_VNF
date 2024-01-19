import numpy as np
import copy
import os
from tqdm import tqdm
import time
import networkx

from ..graph import Network, Link, Node, SFC_SET, SFC

class Wolf():
    def __init__(self, id, network: Network, sfc: SFC, cfg, num_sfc: int, vector=None):
        self.id = id
        self.network = copy.deepcopy(network)
        self.sfc = sfc
        self.cfg = cfg
        self.num_sfc = num_sfc
        self.fitness = None
        self.feasible = False
        self.debug = self.cfg.debug

        # initialize
        self.source_id = sfc.source
        self.destination_id = sfc.destination
        self.path = []
        self.len_vector = len(self.sfc.vnf_list)
        self.vector = vector if vector is not None else np.random.choice(self.network.server_ids, self.len_vector)

    def cal_fitness(self, forced=False):
        if self.fitness and forced is False:
            return self.fitness
        
        self.R_cap, self.R_server, self.R_vnf = 1, 1, 1
        self.fitness = 1
        self.path = []

        self.path.append(self.source_id)
        if self.network.N[self.source_id].consume_mem(self.sfc.memory) is False:
            if self.debug: print("Wrong at ", self.source_id)
            return self.fitness

        prev_id = self.source_id
        for id_cnt_vnf, current_id in enumerate(self.vector):
            if prev_id != current_id: # routing
                self.create_feasible_networkx()
                subpath = networkx.shortest_path(self.nx_network, source=prev_id, target=current_id, weight='delay')
                if self.debug: print("Subpath:", subpath)
                for id in range(len(subpath) - 1):
                    _node = self.network.N[subpath[id + 1]]
                    _link = self.network.adj[subpath[id]][subpath[id+1]]
                    if _node.consume_mem(self.sfc.memory) is False:
                        raise Exception
                    if _link.consume(self.sfc.bw) is False:
                        raise Exception
                self.path.extend(subpath[1:])
                prev_id = current_id

            self.path.append(current_id)
            node = self.network.N[current_id]
            # install vnf
            if node.consume_cpu(self.sfc.cpu) and node.install_vnf(self.sfc.vnf_list[id_cnt_vnf]):
                if self.debug: print(f"Wolf {self.id} installs VNF {self.sfc.vnf_list[id_cnt_vnf]} at {current_id}")
            else:
                return self.fitness
            
        if current_id != self.destination_id:
            self.create_feasible_networkx()
            subpath = networkx.shortest_path(self.nx_network, source=current_id, target=self.destination_id, weight='delay')
            if self.debug: print("Subpath:", subpath)     
            for id in range(len(subpath) - 1):
                _node = self.network.N[subpath[id + 1]]
                _link = self.network.adj[subpath[id]][subpath[id+1]]
                if _node.consume_mem(self.sfc.memory) is False:
                    raise Exception
                if _link.consume(self.sfc.bw) is False:
                    raise Exception
            self.path.extend(subpath[1:])

        self.R_cap = (np.sum(np.array([link.total_delay for link in self.network.L.values()])) + np.sum(np.array([server.total_delay for server in self.network.N_server]))) / ((self.network.total_delay_link + self.network.total_delay_server) * self.num_sfc)
        self.R_server = np.sum(np.array([self.network.N[id].cost for id in self.network.server_ids if len(self.network.N[id].vnf_used) > 0])) / self.network.sum_cost_servers
        self.R_vnf = np.sum(np.array([self.network.N[id].total_installed_vnf_cost for id in self.network.server_ids])) / np.sum(self.network.cost_vnfs)
        self.fitness = ((self.cfg.weight_vector[0]* self.R_cap) + (self.cfg.weight_vector[1] * (self.R_server + self.R_vnf)/2))/( self.cfg.weight_vector[0]+ self.cfg.weight_vector[1])
        self.feasible = True
        return self.fitness
    
    def update_sfc(self):
        self.sfc.path = self.path
        self.sfc.vnf_location = self.vector
        self.sfc.finished = True

    def check_feasible_node(self, node: Node, vnf_pass=False, vnf_id=-1):
        if node.type == 1 and vnf_pass:
            if (node.check_num_vnf(vnf_id) is False) or node.cpu_available < self.sfc.cpu:
                return False
        if node.type == 0:
            if node.mem_available < self.sfc.memory: 
                return False
        return True 

    def check_feasible_link(self, link: Link):
        if link.resource_available < self.sfc.bw:
            return False
        return True

    def create_feasible_networkx(self):
        self.nx_network = networkx.Graph()
        for in_node_id in range(self.network.num_nodes):
            if self.check_feasible_node(self.network.N[in_node_id]):
                for out_node_id, link in self.network.adj[in_node_id].items():
                    if self.check_feasible_link(link) and self.check_feasible_node(self.network.N[out_node_id]):
                        self.nx_network.add_edge(in_node_id, out_node_id, delay=link.delay)

    def __repr__(self) -> str:
        return f"Wolf {self.id} | Vector: {self.vector} | Fitness: {self.fitness}"