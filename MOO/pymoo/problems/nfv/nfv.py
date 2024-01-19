import random
import copy
import heapq
import numpy as np

import pymoo.gradient.toolbox as anp
from pymoo.core.problem import Problem

from .graph.sfc_set import SFC_SET
from .graph.sfc import SFC
from .graph.network import Network
from .individual import Individual
class NFV(Problem):
    def __init__(self, network: Network, sfc_set: SFC_SET, **kwargs):
        self.network = network
        self.sfcs = sfc_set
        self.num_server = self.network.num_servers
        self.vnf_max = self.network.num_vnfs_limit
        self.num_vnf = self.network.num_type_vnfs
        n_var = self.network.num_vnfs_limit * self.network.num_servers
        super().__init__(n_var = n_var, n_obj=2, xl=0, xu=1, vtype=int, **kwargs)
        
    def generate_individual(self):
        individual = Individual()
        individual.features = [random.uniform(0, 1) for _ in range(self.network.num_vnfs_limit * self.network.num_servers)]
        individual.check_use = [0] * len(individual.features)
        return individual
    
    def decode(self, individual):
        decode_ind = []
        # print("indi", individual.features)
        temp = sorted(individual.features)
        for x in individual.features:
            idx = temp.index(x)
            decode_ind.append(idx+1)
            temp[idx] = -1
            decode_ind = [(int(x%(self.network.num_type_vnfs + 1))) for x in decode_ind]
        individual.new_features = decode_ind
    
    def _evaluate(self, x, out, *args, **kwargs):
        f1 = []
        f2 = []
        
        for i in range(len(x)):
            individual = Individual()
            individual.features = x[i]
            individual.check_use = [0] * len(individual.features)
            self.decode(individual)
            # print("individual new features:", individual.new_features)
            network_copy = copy.deepcopy(self.network)
            sfcs_copy = copy.deepcopy(self.sfcs)
            check = True
            for sfc in sfcs_copy.sfc_set:
                if check == False:
                    break
                for i in sfc.vnf_list:
                    ser, pos = self.find_servers_have_vnf_i(i, individual, network_copy)
                    if len(ser) == 0:
                        check = False
                        individual.objectives = [float('inf') for _ in range(2) ]
                        break
                    ser = self.remove_servers_invalid(ser, sfc, network_copy)
                    self.find_path_dijkstra(ser, network_copy, sfc, pos, individual)
                if check == False:
                    break    
                self.find_path_dijkstra([network_copy.N[sfc.destination]], network_copy, sfc, None, individual)    
            # print("individual.features", individual.features) 
            self._kichhoatNodes(individual, network_copy)
            individual.objectives = self._obj_func(network_copy, self.sfcs)
            f1.append(individual.objectives[0])
            f2.append(individual.objectives[1])
            # f3.append(individual.objectives[2])
        out["F"] = anp.column_stack([f1, f2])
            
    def _kichhoatNodes(self, individual: Individual, network_copy: Network) -> None:
        for i in range(len(individual.features)):
            if individual.check_use[i] == 0:
                individual.new_features[i] = 10
        # print("individual.features: ", individual.features)
        for node_server_id in range(self.network.num_servers):
            server_id = network_copy.server_ids[node_server_id]
            test_server_empty = True
            for vnf_id in range(0, self.network.num_vnfs_limit):
                index_vnf = node_server_id * network_copy.num_vnfs_limit + vnf_id - 1
                vnf = individual.new_features[index_vnf]
                # print("vnf: ", vnf)
                #FIXME - có 10 vnf thì vnf0 tới vnf9
                if vnf != 10:
                    # print("vnf_id", vnf_id)
                    network_copy.cost_vnfs_use += network_copy.N[server_id].vnf_cost[vnf-1]
                    test_server_empty = False
            
            if test_server_empty == False:
                network_copy.cost_servers_use += network_copy.N[server_id].cost
                        
    def _obj_func(self, network_copy: Network, sfc_set: SFC_SET):
        fitness = []
        # delay of all sfcs
        fitness.append((network_copy.delay_link + network_copy.delay_server)/((network_copy.total_delay_link + network_copy.total_delay_server)*sfc_set.num_sfc))
        # cost of install servers and install vnfs
        fitness.append(0.5 * (network_copy.cost_servers_use/network_copy.sum_cost_servers + network_copy.cost_vnfs_use/network_copy.max_cost_vnfs))
        # cost of install vnfs
        # fitness.append(network_copy.cost_vnfs_use/network_copy.max_cost_vnfs)
        return fitness
    
    # return list of Node (class)
    def find_servers_have_vnf_i(self, i: int, individual: Individual, network_copy: Network):
        i = int(i)
        positions = [index for index, value in enumerate(individual.new_features) if value == i]
        # print("positions: ", positions)
        server_id = [positions[index] // self.network.num_vnfs_limit  for index in range(len(positions))]
        # print("server_id: ", server_id)
        # chuyển id đó sang node đó
        server_node = [network_copy.N[i] for i in server_id]
        for i, s in enumerate(server_node):
            server_node[i] = network_copy.N[network_copy.server_ids[server_node[i].id]]
        pos = {node: positions[index] for index, node in enumerate(server_node)}
        return server_node, pos
        
    def remove_servers_invalid(self, ser, sfc: SFC, network_copy: Network):
        valid_ser = []
        for server_node in ser:
            if sfc.cpu < server_node.cpu_available:
                valid_ser.append(server_node)
        return valid_ser
    # find path have min bandwidth, then update constraint    
    def find_path_dijkstra(self, end_nodes, network_copy: Network, sfc: SFC, pos, individual: Individual):
        paths = {}
        distances = {node: float('inf') for node in network_copy.N.values()}
        start_id = sfc.path[-1] 
        # print("start_id", start_id)
        distances[network_copy.N[start_id]] = 0
        
        # for end_node in end_nodes:
            # print("end node: ", end_node.id)
        # time.sleep(10)
        previous_nodes = {node: None for node in network_copy.N.values()} 
        unvisited_nodes = [(0, sfc.path[-1])]
        while unvisited_nodes:
            current_distance, current_id = heapq.heappop(unvisited_nodes)
            current_node = network_copy.N[current_id]
            if current_distance > distances[current_node]:
                continue 
            for neighbor_id, link in network_copy.adj[current_id].items():
                weight = link.delay
                # print("weight: ", weight)
                neighbor_node = network_copy.N[neighbor_id]
                # print("current_distance: ", current_distance)
                distance = current_distance + weight
                # print("distance: ", distance)
                
                # checking constraint
                if sfc.bw > link.bw_available:
                    continue
                if neighbor_node.type == True and sfc.cpu > network_copy.N[neighbor_id].cpu_available:
                    continue
                if neighbor_node.type == False and sfc.memory > network_copy.N[neighbor_id].mem_available:
                    continue
                
                # print("------distances[neighbor] before: ", distances[neighbor_node])
                if distance < distances[neighbor_node]:
                    distances[neighbor_node] = distance
                    # print("------distances[neighbor] after: ", distances[neighbor_node])
                    previous_nodes[neighbor_node] = current_node
                    heapq.heappush(unvisited_nodes, (distance, neighbor_id))
        for end_node in end_nodes:
            # print("end node: ", end_node)
            path = [end_node]
            temp_node = end_node
            # print("temp_node.id: ", temp_node.id)
            # print("startiddd: ", start_id)
            while temp_node.id != start_id:
                temp_node = previous_nodes[temp_node]
                if temp_node is None:
                    break
                # print("temp_node id: ", temp_node. id)
                path.append(temp_node)
            path.reverse()
            paths[end_node] = [path, distances[end_node]]
    
        smallest_weight = float('inf')
        smallest_end_node = None
        for end_node in end_nodes:
            if distances[end_node] < smallest_weight:
                smallest_weight = distances[end_node]
                smallest_end_node = end_node
        # check using
        # print("smallest_end_node: ", smallest_end_node.id)
        
        if pos is not None:
            # print("positions: ", pos[smallest_end_node])
            individual.check_use[pos[smallest_end_node]] = 1
        
        path_id = []
        for node in (paths[smallest_end_node][0]):
            path_id.append(node.id)
        end_before = sfc.path[-1] 
        sfc.path.pop()  
        sfc.path.extend(path_id)
        # print("sfc.path: ", sfc.path)
        smallest_end_node.cpu_available -= sfc.cpu
        network_copy.delay_link += paths[smallest_end_node][1]
        for i, node in enumerate(paths[smallest_end_node][0]):
            
        #     if i == 0:
        #         print("type end, node: ", type(end_before), type(node))
        #         print("end, node", end_before, node.id)
        #         network_copy.delay_link += network_copy.adj[end_before][node.id].delay
        #     elif (i+1) < len(paths[smallest_end_node][0]):
        #         network_copy.delay_link += network_copy.adj[node.id][paths[smallest_end_node][0][i+1].id].delay
            # network_copy.adj[current.id][previous_nodes[current].id].bw_available -= sfc.bw  
            if node.type == False:
                node.mem_available -= sfc.memory
            else:
                node.cpu_available -= sfc.cpu  
    
    # check vnf not used and then calculate again the fitness of individual
