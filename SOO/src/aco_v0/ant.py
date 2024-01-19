from ..graph import *

import networkx
import copy
import random

# for new formulation
class AntV0():
    def __init__(self, id, network: Network, sfc: SFC, num_sfc: int):
        self.id = id
        self.network = copy.deepcopy(network)
        self.sfc = sfc
        self.num_sfc = num_sfc

        self.source_id = self.sfc.source
        self.destination_id = self.sfc.destination
        self.path = []
        self.vnf_location = []
        self.vnf_demand = copy.copy(self.sfc.vnf_list)

        self.fitness = -1
        self.num_actived_servers = -1

    def run(self, cfg):
        self.cfg = cfg
        # set up parameters
        self.Q = cfg.Q
        self.threshold_state_lower = cfg.threshold_state_lower
        self.threshold_state_upper = cfg.threshold_state_upper
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.debug = cfg.debug
        self.if_weight_subpath = cfg.if_weight_subpath

        current_id = self.source_id
        self.path = []
        self.path.append(current_id)
        self.finished = False

        if self.debug:
            print("="*20)
            print(f"Ant {self.id} start from {current_id} to {self.destination_id} ...")

        current_node = self.network.N[current_id]
        if current_node.consume_mem(self.sfc.memory) is False: return
        
        # install vnf
        if current_node.type == 1:
            while(True):
                if not self.vnf_demand:
                    break
                current_vnf = self.vnf_demand[0]
                if self.check_feasible_node(node=current_node, vnf_pass=True, vnf_id=current_vnf):
                    if self.install_vnf(current_node, current_vnf) is False:
                        raise Exception
                else:
                    break

        while current_id != self.destination_id:
            # goes to the destination
            if not self.vnf_demand:
                self.create_feasible_networkx(self.network.N[current_id])
                path = networkx.shortest_path(self.nx_network, source=current_id, target=self.destination_id)
                for id in range(len(path) - 1):
                    _node = self.network.N[path[id + 1]]
                    _link = self.network.adj[path[id]][path[id+1]]
                    if _node.consume_mem(self.sfc.memory) is False:
                        raise Exception
                    if _link.consume(self.sfc.bw) is False:
                        raise Exception
                self.path.extend(path[1:])
                break

            current_node = self.network.N[current_id]
            # link selection
            next_subpath = self.link_selection(current_node)
            if next_subpath is False:
                break
            current_id = next_subpath[-1]
            if self.debug: print()

        if self.debug: print("Final path: ", self.path)
        if len(self.vnf_demand) == 0:
            self.finished = True
        
        self.compute_fitness()
        self.count_actived_servers()

    def link_selection(self, node: Node):
        if(self.debug): print(f"Subpath selection at {node.id}", end=': ')

        if self.vnf_demand:
            current_vnf = self.vnf_demand[0]
            candidate_servers_id = [sids for sids in self.network.server_ids
                                        if sids != node.id 
                                        and self.check_feasible_node(self.network.N[sids], vnf_pass=True, neighbor_check=True, vnf_id=current_vnf)]
        else:
            # self.keypoint_consume[self.destination_id] -= self.sfc.memory
            if self.check_feasible_node(self.network.N[self.destination_id]):
                candidate_servers_id = [self.destination_id]
            else:
                if(self.debug): print("Destination node is infeasible!")
                # print(self.network.N[self.destination_id], self.keypoint_consume[self.destination_id])
                return False

        subpaths = self.generate_candidate_subpath(node, candidate_servers_id, self.cfg.num_dijkstra_paths)
        if(len(subpaths) == 0):
            if(self.debug): print(f"Ant {self.id}|There is no subpaths at {node.id} go to {candidate_servers_id}")
            return False
        next_subpath = -1  

        # build criteria
        subpaths_bw = []
        subpaths_delay = []
        subpaths_pheromone = []
        for subpath in subpaths:
            subpaths_bw.append(np.mean(np.array([self.network.adj[subpath[i]][subpath[i+1]].resource_available for i in range(len(subpath) - 1)])))
            subpaths_delay.append(np.sum(np.array([self.network.adj[subpath[i]][subpath[i+1]].delay for i in range(len(subpath) - 1)])) + self.network.N[subpath[-1]].delay)
            subpaths_pheromone.append(np.mean(np.array([self.network.pheromone[subpath[i]][subpath[i+1]] for i in range(len(subpath) - 1)])))
        
        # criteria 1: about whether vnf is installed or not
        if self.vnf_demand:
            if1 = np.array([np.random.uniform(low=self.threshold_state_upper, high=1) if current_vnf in self.network.N[subpath[-1]].vnf_used
                            else self.threshold_state_lower*self.network.min_cost_vnfs_axis_server[current_vnf]/self.network.N[subpath[-1]].vnf_cost[current_vnf]
                            for subpath in subpaths])
            for subpath in subpaths:
                if self.network.min_cost_vnfs_axis_server[current_vnf]/self.network.N[subpath[-1]].vnf_cost[current_vnf] > 1: print("Code ngu 3")
        else:
            if1 = np.ones(len(subpaths))
        if1 = if1 / np.sum(if1)
        
        # criteria 2: about state (except path go to the destination)
        if self.vnf_demand:
            if2 = np.array([np.random.uniform(low=self.threshold_state_upper, high=1) if len(self.network.N[subpath[-1]].vnf_used) > 0
                            else self.threshold_state_lower*self.network.min_cost_servers/self.network.N[subpath[-1]].cost
                            for subpath in subpaths])   
            for subpath in subpaths:
                if self.network.min_cost_servers/self.network.N[subpath[-1]].cost > 1: print("Code ngu 4")
        else:
            if2 = np.ones(len(subpaths))
        if2 = if2 / np.sum(if2)

        # criteria 3: about total delay of subpath
        if3 = 1 / np.array(subpaths_delay)
        if3 = if3 / np.sum(if3)

        # criteria 4: about mean link's bandwidth
        if4 = np.array(subpaths_bw)
        if4 = if4 / np.sum(if4)
        
        if_weight = self.if_weight_subpath / np.sum(self.if_weight_subpath)
        inspiration_factors = if_weight[0]*if1 + if_weight[1]*if2 + if_weight[2]*if3 # + if_weight[3]*if4

        pheromones = np.array(subpaths_pheromone)
        pheromones = pheromones / np.sum(pheromones)

        probs = inspiration_factors**self.alpha * pheromones**self.beta
        probs = probs / np.sum(probs)

        next_subpath_id = np.random.choice(np.arange(len(subpaths)), p=probs)   
        next_subpath = subpaths[next_subpath_id]

        # update network
        for i in range(len(next_subpath) - 1): # server_consume1
            link = self.network.adj[next_subpath[i]][next_subpath[i+1]]
            if link.consume(self.sfc.bw) is False:
                print(f"Ant {self.id}|Something is wrong 1 at {link}!", self.check_feasible_link(link))
                raise Exception("Error")
                return False

            node = self.network.N[next_subpath[i+1]] 
            if node.consume_mem(self.sfc.memory) is False:
                print(f"Ant {self.id}|Something is wrong 2 at {node}!", self.check_feasible_node(node))
                raise Exception("Error")
                return False

        if(self.debug): print(next_subpath, end=',')

        while(True):
            if not self.vnf_demand:
                break
            # install vnf 
            node = self.network.N[next_subpath[-1]]
            current_vnf = self.vnf_demand[0]
            if self.check_feasible_node(node=node, vnf_pass=True, vnf_id=current_vnf):
                if node.consume_cpu(self.sfc.cpu) and node.install_vnf(current_vnf):
                    if self.debug: print(f"Ant {self.id}|Install VNF {current_vnf} at node {node.id}")
                    # update vnf-related
                    self.vnf_location.append(node.id)
                    self.vnf_demand.pop(0)
                    next_subpath.append(next_subpath[-1])
                else:
                    raise Exception
            else:
                break

        self.path.extend(next_subpath[1:])
        return next_subpath
        
    def compute_fitness(self):
        if self.finished is False: # penalty
            self.R_cap, self.R_server, self.R_vnf = 1, 1, 1
        else:
            total_delay_link = np.sum(np.array([link.total_delay for link in self.network.L.values()]))
            total_delay_server = np.sum(np.array([server.total_delay for server in self.network.N_server]))
            self.R_cap = (total_delay_link + total_delay_server) / ((self.network.total_delay_link + self.network.total_delay_server) * self.num_sfc)
            self.R_server = np.sum(np.array([self.network.N[id].cost for id in self.network.server_ids if len(self.network.N[id].vnf_used) > 0])) / self.network.sum_cost_servers
            self.R_vnf = np.sum(np.array([self.network.N[id].total_installed_vnf_cost for id in self.network.server_ids])) / np.sum(self.network.cost_vnfs)
        self.objective_value = self.R_cap + self.R_server + self.R_vnf
        self.fitness = 1 / self.objective_value 

    def count_actived_servers(self):
        self.num_actived_servers = self.network.count_actived_servers() if self.finished else -1

    def find_feasible_neighbor(self, node: Node):
        neighbor_nodes, links = self.network.find_all_neighbor(node)
        feasible_nodes, feasible_links = [], []
        for id, (_node, _link) in enumerate(zip(neighbor_nodes, links)):
            if _node.type == 0 and _node.id in self.path:
                continue
            if self.check_feasible_link(_link) and self.check_feasible_node(_node):
                feasible_links.append(_link)
                feasible_nodes.append(_node)
        return feasible_nodes, feasible_links

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

    def install_vnf(self, node: Node, current_vnf: VNF):
        # update network: conflict when changing formulation XX!
        if node.consume_cpu(self.sfc.cpu) and node.install_vnf(current_vnf):
            if self.debug: print(f"Install VNF {current_vnf} at node {node.id}")
        else:
            if(self.debug): print("Something doesn't work!")
            raise Exception

        # update vnf-related
        self.vnf_location.append(node.id)
        self.vnf_demand.pop(0)

        # update path
        self.path.append(node.id)
        return True

    def generate_candidate_subpath(self, node: Node, candidate_servers_id, k_paths_per_candidate=1):
        subpaths = []
        self.create_feasible_networkx(node)
        self.weight_link = "delay"# if self.network.num_nodes <= 20 else None
        for next_id in candidate_servers_id:
            # try:
            # if self.nx_network.has_node(node.id) and networkx.has_path(self.nx_network, node.id, next_id):
            if k_paths_per_candidate == 1:
                subpaths.append(networkx.shortest_path(self.nx_network, node.id, next_id, weight=self.weight_link))
            else:
                for cnt, path in enumerate(networkx.shortest_simple_paths(self.nx_network, node.id, next_id, weight=self.weight_link)):
                    if cnt < k_paths_per_candidate:
                        subpaths.append(path)
                    else:
                        break
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

    def __repr__(self) -> str:
        return f'Ant {self.id} | SFC {self.sfc.id} | Route: {self.path}'