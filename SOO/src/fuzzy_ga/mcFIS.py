from ..graph import *
from .fuzzy_system import fuzzy_system
from utils import next_path

import networkx
import random
import copy
import time
import os

NUM_MF_INPUT = 3

class mcFIS():
    def __init__(self, network: Network, sfc_set: SFC_SET, cfg):
        self.network = copy.deepcopy(network)
        self.sfc_set = copy.deepcopy(sfc_set)
        self.cfg = cfg
        self.num_sfc = len(self.sfc_set)
        self.sucessed_sfc = []

    def run(self, A, B):
        assert len(A.shape) == 1 and len(B.shape) == 1
        assert A.shape[0] == 5 and B.shape[0] == 5

        self.A = A
        self.B = B

        rules = np.zeros((NUM_MF_INPUT,)*5) # 5 is dim of input
        for i0 in range(NUM_MF_INPUT):
            for i1 in range(NUM_MF_INPUT):
                for i2 in range(NUM_MF_INPUT):
                    for i3 in range(NUM_MF_INPUT):
                        for i4 in range(NUM_MF_INPUT):
                            rules[i0][i1][i2][i3][i4] = A[0]*(i0+1)**B[0] + A[1]*(i1+1)**B[1] + A[2]*(i2+1)**B[2] + A[3]*(i3+1)**B[3] + A[4]*(i4+1)**B[4]
        rules = np.ceil(rules / np.max(rules) * 7) - 1 # in range [0,6]
        self.rules = rules.astype('int32')
    
        begin = time.time()
        # set_seed(self.cfg.seed) # bug when genes for individuals are the same
        self.num_sucesss_sfc = 0
        if self.cfg.log_path:
            self.cfg.log_path = next_path(self.cfg.log_path+"%s")
            os.makedirs(self.cfg.log_path, exist_ok=True)

        for iter_sfc, self.sfc in enumerate(self.sfc_set.sfc_set):
            self.prev_network = copy.deepcopy(self.network) # reset to previous network in case the current sfc is not satisfied
            if self.cfg.debug: print(f"Service SFC {self.sfc} ...")

            self.current_id = self.source_id = self.sfc.source
            self.destination_id = self.sfc.destination
            self.vnf_demand = self.sfc.vnf_demand
            self.vnf_location = []
            self.feasible = True
            self.path = [self.current_id]

            current_node = self.network.N[self.current_id]
            if current_node.consume_mem(self.sfc.memory) is False:
                raise Exception
            while(len(self.vnf_demand) != 0):
                current_node = self.network.N[self.current_id]
                current_vnf = self.vnf_demand[0]
                next_id = self.placement(current_node)
                if next_id == -1:
                    self.feasible = False
                    break
                elif next_id != self.current_id:
                    self.routing(self.current_id, next_id)
                               
                # install vnf
                self.current_id = next_id
                current_node = self.network.N[self.current_id]
                if current_node.consume_cpu(self.sfc.cpu) and current_node.install_vnf(current_vnf):
                    self.vnf_location.append(current_node.id)
                    self.vnf_demand.pop(0)
                    self.path.append(current_node.id)                   
                else: raise Exception

            if self.current_id != self.destination_id and self.feasible:
                self.routing(self.current_id, self.destination_id)
                self.current_id = self.destination_id

            # update network
            if self.cfg.log_path:
                info = "="*20 + "\n" + f"SFC {str(self.sfc)} - vnf location: {self.vnf_location} path: {self.path} \n Number of servers: {self.network.count_actived_servers()} \n Server: {self.network.N_server} \n Node: {self.network.N} \n Link: {self.network.L} \n"
                with open(f"{self.cfg.log_path}/status.txt", "a") as f:
                    f.write(info)

            if self.cfg.display:
                print(f"SFC {self.sfc.id},feasible: {self.feasible}, vnf location: {self.vnf_location} path: {self.path}, number of active server: {self.network.count_actived_servers()}")
                print("="*20)
            
            if self.feasible:
                self.sucessed_sfc.append(1)
            else:
                self.sucessed_sfc.append(0)

        self.num_sucesss_sfc = np.sum(self.sucessed_sfc)
        self.compute_fitness()
        self.num_actived_servers = self.network.count_actived_servers()
        self.total_time = time.time() - begin
        if self.cfg.log_path:
            info = f"Objective value: {self.objective}({self.R_cap},{self.R_server},{self.R_vnf}) | Total time: {self.total_time}"
            with open(f"{self.cfg.log_path}/status.txt", "a") as f:
                f.write(info)

        if self.cfg.display:
            info = f"Objective value: {self.objective}({self.R_cap},{self.R_server},{self.R_vnf}) | Total time: {self.total_time}"
            print(info)

        
    def compute_fitness(self):
        if self.num_sucesss_sfc == self.num_sfc:
            self.R_cap = (np.sum(np.array([link.total_delay for link in self.network.L.values()])) + np.sum(np.array([server.total_delay for server in self.network.N_server]))) / ((self.network.total_delay_link + self.network.total_delay_server) * self.num_sfc)
            self.R_server = np.sum(np.array([self.network.N[id].cost for id in self.network.server_ids if len(self.network.N[id].vnf_used) > 0])) / self.network.sum_cost_servers
            self.R_vnf = np.sum(np.array([self.network.N[id].total_installed_vnf_cost for id in self.network.server_ids])) / np.sum(self.network.cost_vnfs)
            self.objective = ((self.R_cap*self.cfg.weight_vector[0]) + (self.cfg.weight_vector[1]*(self.R_server + self.R_vnf)/2))/(self.cfg.weight_vector[1]+self.cfg.weight_vector[2])
        else:
            self.R_cap, self.R_server, self.R_vnf = 1,1,1
            self.objective = 1

    def save_result(self, path, version_name):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, version_name), 'a') as f:
            f.write(f"{self.network.name},{self.sfc_set.name},{self.R_cap},{self.R_server},{self.R_vnf},{self.objective},{self.num_sucesss_sfc},{self.num_actived_servers},{self.total_time} \n")

    def routing(self, current_id, next_id):
        self.create_feasible_networkx(self.network.N[current_id])
        next_subpath = networkx.shortest_path(self.nx_network, source=current_id, target=next_id, weight='delay')

        for i in range(len(next_subpath) - 1):
            link = self.network.adj[next_subpath[i]][next_subpath[i+1]]
            if link.consume(self.sfc.bw) is False:
                raise Exception

            node = self.network.N[next_subpath[i+1]] 
            if node.consume_mem(self.sfc.memory) is False:
                raise Exception
        self.path.extend(next_subpath[1:])

    def placement(self, current_node: Node):
        current_vnf = self.vnf_demand[0]
        
        candidate_servers = [
            server for server in self.network.N_server 
            if self.check_feasible_node(server, vnf_pass=True, neighbor_check=True, vnf_id=current_vnf)
        ]
        if len(candidate_servers) == 0:
            # print(f"{self.network.name}-{self.sfc_set.name}: No feasible servers | Genes: {self.A}-{self.B}")
            # print(f"{self.network.N_server}")
            return -1
        candidate_servers_id = [server.id for server in candidate_servers]
        # the active state of server
        fp1 = np.array([1 if len(server.vnf_used) > 0 else 0
                        for server in candidate_servers])
        
        # the cost of server
        fp2 = np.array([self.network.min_cost_servers / server.cost
                        for server in candidate_servers])
        
        # the delay of server
        fp3 = np.array([self.network.min_delay_server / server.delay 
                            for server in candidate_servers])

        # the installed vnf state
        fp4 = np.array([1 if current_vnf in server.vnf_used else 0
                        for server in candidate_servers])
        
        # the distance between current_node -> server
        paths = self.generate_candidate_subpath(current_node, candidate_servers_id, k_paths_per_candidate=1)
        fp5 = np.array([(1- len(path)/self.network.num_nodes)
                        for path in paths])
        
        fp = np.array([fp1,fp2,fp3,fp4,fp5]) # 5 x num_servers
        pf = [fuzzy_system(fp[:, i], self.rules) for i in range(len(candidate_servers))]

        next_id = candidate_servers_id[np.argmax(pf)]
        return next_id        
        
    def check_feasible_node(self, node: Node, vnf_pass=False, neighbor_check=False, saving_src_check=True, vnf_id=-1):
        if node.type == 1 and vnf_pass: #server_consume1
            if (node.check_num_vnf(vnf_id) is False) or node.cpu_available < self.sfc.cpu:
                return False

        if node.type == 0:
            # resource_saving = self.keypoint_consume[node.id] if node.id in self.keypoint_consume.keys() and saving_src_check else 0
            if node.mem_available < self.sfc.memory: 
                return False

        # if neighbor_check: # exist at least one possible path to go when we are at "node"
        #     flag = False
        #     for out_node_id, link in self.network.adj[node.id].items():
        #         if self.check_feasible_node(self.network.N[out_node_id]) and self.check_feasible_link(link):
        #             flag = True
        #             break
        #     if(flag is False): return False
        return True 

    def check_feasible_link(self, link: Link):
        if link.resource_available < self.sfc.bw:
            return False
        return True

    def generate_candidate_subpath(self, node: Node, candidate_servers_id, k_paths_per_candidate=1):
        subpaths = []
        self.create_feasible_networkx(node)
        for next_id in candidate_servers_id:
            # try:
            # if self.nx_network.has_node(node.id) and networkx.has_path(self.nx_network, node.id, next_id):
            if k_paths_per_candidate > 1:
                for cnt, path in enumerate(networkx.shortest_simple_paths(self.nx_network, node.id, next_id)):
                    if cnt < k_paths_per_candidate:
                        subpaths.append(path)
                    else:
                        break
            else:
                path = networkx.shortest_path(self.nx_network, source=node.id, target=next_id, weight='delay')
                subpaths.append(path)
            # except Exception as e:
            #     print(e)
            #     print(f"Ant {self.id}|current node: {node.id} - candidate: {next_id}-{self.check_feasible_node(self.network.N[next_id], neighbor_check=True)}")
            #     if self.nx_network.has_node(node.id) is False:
            #         print(f"Check feasible:{self.check_feasible_node(self.network.N[node.id], neighbor_check=True)}")
            #         for _id, _link in self.network.adj[node.id].items():
            #             print(self.network.N[_id], _link)
            #     else:
            #         print(f"{self.network.adj[next_id]}")
            #     print(self.path)
            #     return subpaths
        return subpaths

    def create_feasible_networkx(self, current_node: Node):
        self.nx_network = networkx.Graph()
        for in_node_id in range(self.network.num_nodes):
            if self.check_feasible_node(self.network.N[in_node_id]) or in_node_id is current_node.id:
                for out_node_id, link in self.network.adj[in_node_id].items():
                    if self.check_feasible_link(link) and self.check_feasible_node(self.network.N[out_node_id]):
                        self.nx_network.add_edge(in_node_id, out_node_id, delay=link.delay)
            
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)