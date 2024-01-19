import numpy as np
import copy
import networkx
import matplotlib.pyplot as plt

from .link import *
from .node import *
from .vnf import *

class Network():
    
    def __init__(self, input_path=None, undirected=True) -> None:
        self.input_path = input_path
        self.undirected = undirected
        self.name = self.input_path.split("/")[-2]

        #print(f"Initialize network from: {self.input_path} ...")
        with open(self.input_path, "r") as f:
            lines = f.read().splitlines()

        self.N = dict()
        self.L = dict()
        self.adj = dict()#adjacent
        self.num_nodes = 0
        self.num_links = 0        
        self.num_servers = 0
        self.delay_link = 0
        self.total_delay_link = 0
        self.delay_server = 0
        self.total_delay_server = 0
        self.cost_servers_use = 0
        self.cost_servers = []
        self.cost_vnfs_use = 0
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
        print("sum server cost: ", self.sum_cost_servers)
        self.min_cost_servers = np.min(self.cost_servers)
        self.max_cost_servers = np.max(self.cost_servers)
        self.N_server = [self.N[id] for id in self.server_ids]

        self.cost_vnfs = np.array(self.cost_vnfs)
        self.min_cost_vnfs_axis_server = np.min(self.cost_vnfs, axis = 0)
        self.min_delay_local_server = self.N_server[0].delay
        for server in self.N_server:
            self.min_delay_local_server = min(self.min_delay_local_server, server.delay)

        self.max_delay_server = self.N_server[0].delay
        for server in self.N_server:
            self.max_delay_server = max(self.max_delay_server, server.delay)

        num_links = int(lines[2+self.num_nodes])
        for id in range(3 + num_nodes, 3 + num_nodes + num_links):
            line = lines[id].strip().split()
            _source_id, _destination_id, _delay = int(line[0]), int(line[1]), int(line[2])
            self.add_link(Link(source =self.N[_source_id], 
                                destination= self.N[_destination_id], delay=_delay))
            self.total_delay_link += _delay

        self.create_networkx()

        self.max_cost_vnfs = sum(i.total_vnf_cost for i in self.N_server)
        print(self.max_cost_vnfs)
        self.max_delay_links = sum(i.delay for i in self.L.values())


    def create_networkx(self):
        self.nx_network = networkx.Graph()
        adj_id = []

        for in_node_id in range(self.num_nodes):
            for out_node_id, link in self.adj[in_node_id].items():
                adj_id.append([link.source.id, link.destination.id])
        self.nx_network.add_edges_from(adj_id)

    def _update_adjacent(self, link):
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
            self._update_adjacent(link)
            self.num_links += 1
    
    def find_server(self, server_id):
        return self.server_ids.index(server_id)