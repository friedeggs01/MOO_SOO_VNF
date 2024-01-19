import ast
import numpy as np
import copy
import networkx
import os, glob, shutil
import matplotlib.pyplot as plt

from .link import *
from .node import *
from .vnf import *

# for new formulation
class Network():
    def __init__(self, V=dict(), L=dict(), input_path=None, undirected=True) -> None:
        self.input_path = input_path
        self.undirected = undirected
        self.name = self.input_path.split("/")[-2]

        #print(f"Initialize network from: {self.input_path} ...")
        with open(self.input_path, "r") as f:
            lines = f.read().splitlines()

        self.N = dict()
        self.L = dict()
        self.adj = dict()
        self.num_nodes = 0
        self.num_links = 0        
        self.num_servers = 0
        self.total_delay_link = 0
        self.total_delay_server = 0
        self.cost_servers = []
        self.cost_vnfs = []
        self.snnode_ids = []
        self.server_ids = []

        line = list(map(int, lines[0].strip().split()))
        if len(line) == 2:
            self.num_type_vnfs, self.num_vnfs_limit = line
        else:
            self.num_type_vnfs, self.num_vnfs_limit = line[0], line[0]
        num_nodes = int(lines[1])
        for id in range(2, 2 + num_nodes):
            line = lines[id].strip().split()
            line = [int(l) for l in line]
            _id, _delay, _cost = line[0], line[1], line[2]
            if _cost == -1:
                self.add_node(Node(id=_id, type=0, delay=_delay, cost=_cost))
                self.snnode_ids.append(_id)
            else:
                self.add_node(Node(id=_id, type=1, delay=_delay, cost=_cost,
                            vnf_cost=line[3:], vnf_possible=np.arange(self.num_type_vnfs), vnf_used=[], num_vnfs_limit = self.num_vnfs_limit))
                self.server_ids.append(_id)
                self.cost_servers.append(_cost)
                self.cost_vnfs.append(line[3:])
                self.total_delay_server += _delay

        self.num_servers = len(self.server_ids)
        self.sum_cost_servers = np.sum(self.cost_servers)
        self.min_cost_servers = np.min(self.cost_servers)
        self.max_cost_servers = np.max(self.cost_servers)
        self.N_server = [self.N[id] for id in self.server_ids]

        self.cost_vnfs = np.array(self.cost_vnfs)
        self.min_cost_vnfs_axis_server = np.min(self.cost_vnfs, axis = 0)
        self.min_delay_server = self.N_server[0].delay
        for server in self.N_server:
            self.min_delay_server = min(self.min_delay_server, server.delay)

        num_links = int(lines[2+self.num_nodes])
        for id in range(3 + num_nodes, 3 + num_nodes + num_links):
            line = lines[id].strip().split()
            _source_id, _destination_id, _delay = int(line[0]), int(line[1]), int(line[2])
            self.add_link(Link(source =self.N[_source_id], 
                                destination= self.N[_destination_id], delay=_delay))
            self.total_delay_link += _delay
        
        #print(f"Initialized: {self.__repr__()}")

        self.create_networkx()

    def create_networkx(self):
        self.nx_network = networkx.Graph()
        adj_id = []

        for in_node_id in range(self.num_nodes):
            for out_node_id, link in self.adj[in_node_id].items():
                adj_id.append([link.source.id, link.destination.id])
        self.nx_network.add_edges_from(adj_id)

        self.nx_network_pos = networkx.spring_layout(self.nx_network)

    def update_adjacent(self, link):
        source, destination = link.source, link.destination
        if source.id in self.adj.keys():
            self.adj[source.id][destination.id] = link
        else:
            self.adj[source.id] = dict()
            self.adj[source.id][destination.id] = link
 
        if self.undirected: 
            if destination.id in self.adj.keys():
                self.adj[destination.id][source.id] = link
            else:
                self.adj[destination.id] = dict()
                self.adj[destination.id][source.id] = link

    def add_node(self, node: Node) -> None:
        if node.id in self.N.keys():
            print("ID node is existed!")
        else:
            self.N[node.id] = node
            self.num_nodes += 1

    def add_link(self, link: Link):
        if link.id in self.N.keys():
            print("ID link is existed!")
        else:
            self.L[link.id] = link
            self.update_adjacent(link)
            self.num_links += 1

    def delete_node(self):
        pass

    def delete_link(self):
        pass

    def build_pheromone(self, num_sfcs = -1):
        self.pheromone = np.zeros((self.num_nodes, self.num_nodes))
        # for links
        for u, adj_u in self.adj.items():
            for v in adj_u.keys():
                self.pheromone[u][v] = 1
        # for nodes
        for node in self.N.values():
            if node.type:
                self.pheromone[node.id][node.id] = 1

        if num_sfcs != -1:
            self.pheromone = np.stack([self.pheromone]*num_sfcs, axis=0)

    def create_constraints(self, sfc_set):
        self.memmax = sfc_set.memmax
        self.cpumax = sfc_set.cpumax
        self.capmax = sfc_set.capmax

        for node in self.N.values():
            node.mem_capacity, node.mem_available = sfc_set.memmax, sfc_set.memmax
            if node.type == 1:
                node.cpu_capacity, node.cpu_available = sfc_set.cpumax, sfc_set.cpumax

        for link in self.L.values():
            link.resource_capacity, link.resource_available = sfc_set.capmax, sfc_set.capmax

    def __repr__(self) -> str:
        return "Network {}| No.Node: {} | No.Link: {} | No. type VNF: {}".format(self.name, self.num_nodes, self.num_links, self.num_type_vnfs)

    def count_actived_servers(self, vnf_id=None):
        cnt = 0
        if vnf_id is None:
            for node in self.N_server:
                if len(node.vnf_used): cnt += 1
        else: # count the number of servers which are activated and able to install vnf_id
            for node in self.N_server:
                if len(node.vnf_used) and vnf_id in node.vnf_possible: cnt += 1
        return cnt
    
    def count_actived_servers_installed_vnf(self, vnf_id):
        cnt = 0
        for node in self.N_server:
            if vnf_id in node.vnf_used: cnt += 1
        return cnt

    def find_all_neighbor_by_id(self, id):
        neighbor = [x for x in self.adj[id].keys()]
        return neighbor

    def find_all_neighbor(self, node):
        neighbor = list(self.adj[node.id].keys())
        neighbor = [self.N[x] for x in neighbor]
        link = list(self.adj[node.id].values())
        return neighbor, link

    # Version 2
    def find_all_path(self, source_id: int, destination_id: int, prior_mark: np.array=None, max_hop: int=-1):
        if self.nx_network is None:
            self.create_networkx()
        paths = []
    
        for cnt, path in enumerate(networkx.shortest_simple_paths(self.nx_network, source_id, destination_id)):
            if cnt >= 10 and self.num_nodes > 50:
                break
            if len(path) > max_hop and max_hop != -1:
                break
            paths.append(path)
        return paths

    def dfs(self, current_id: int, destination_id: int, path: list, visited: list, paths: list, lengths: int, prior_mark, max_hop: int=-1):
        if(len(path) > max_hop and max_hop != -1):
            return
        visited[current_id] = True
        path.append(current_id)

        # print(current_id, destination_id)

        if current_id == destination_id:
            lengths.append(len(path))
            paths.append(copy.copy(path))
        else:
            for neighbor_id in self.adj[current_id].keys():
                if visited[neighbor_id] is False:
                    if prior_mark is not None and prior_mark[neighbor_id]:
                        continue
                    self.dfs(neighbor_id, destination_id, path, visited, paths, lengths, prior_mark, max_hop)
        
        path.pop()
        visited[current_id] = False
    
    def compute_min_dist(self, source: int, destination: int=None):
        queue = []
        visited = [False for i in range(self.num_nodes)]
        dist = [-1 for i in range(self.num_nodes)]

        queue.append(source)
        visited[source] = True
        dist[source] = 0

        while(len(queue) > 0):
            u = queue[0]
            queue.pop(0)

            for v in self.adj[u].keys():
                if(visited[v] is False):
                    visited[v] = True
                    dist[v] = dist[u] + 1
                    queue.append(v)

                    if destination == v:
                        return dist[v]
        return dist

    def visualize(self, path=None):
        if self.nx_network is None:
            self.create_networkx()

        networkx.draw_networkx(self.nx_network, pos=self.nx_network_pos)
        plt.title(f"Network | {self.num_nodes} nodes | {self.num_links} links")
        if path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()
