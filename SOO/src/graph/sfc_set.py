import os
import numpy as np
from .network import *
from .sfc import *

class SFC_SET():
    def __init__(self, input_path=None):
        self.input_path = input_path

        #print(f"Initialize sfcs from: {self.input_path} ...", end=' ')
        with open(self.input_path, "r") as f:
            lines = f.read().splitlines()
        self.name = os.path.split(self.input_path)[-1].split(".")[0]
        self.num_sfc = int(lines[0])
        self.sfc_set = []
        for id in range(1, self.num_sfc + 1):
            line = lines[id].strip().split()
            self.sfc_set.append(SFC(id-1, line))
        #print(f"Initialized!")
        
        self.total_required_vnf = 0
        for sfc in self.sfc_set: self.total_required_vnf += len(sfc.vnf_list)
        self.capmax = np.sum(np.array([sfc.bw*(sfc.num_vnfs + 1) for sfc in self.sfc_set]))
        self.memmax = np.sum(np.array([sfc.memory*(sfc.num_vnfs + 1) for sfc in self.sfc_set]))
        self.cpumax = np.sum(np.array([sfc.cpu*sfc.num_vnfs for sfc in self.sfc_set]))

    def create_global_info(self, network: Network):
        self.network_name = network.name
        self.keypoint_consume = dict() # keypoint includes source node + destination node
        for sfc in self.sfc_set:
            value = sfc.memory #if network.N[sfc.source].type == 0 else 0 #server_consume1
            if sfc.source not in self.keypoint_consume.keys():
                self.keypoint_consume[sfc.source] = value
            else:
                self.keypoint_consume[sfc.source] += value

            value = sfc.memory #if network.N[sfc.destination].type == 0 else 0 #server_consume1
            if sfc.destination not in self.keypoint_consume.keys():
                self.keypoint_consume[sfc.destination] = value
            else:
                self.keypoint_consume[sfc.destination] += value

    def sort(self):
        self.sfc_set = sorted(self.sfc_set, key=lambda x: -x.density)

    def total_type_vnf_require(self):
        lst = []
        for sfc in self.sfc_set:
            lst.extend(sfc.vnf_list)
        return set(lst)

    def __len__(self):
        return len(self.sfc_set)

    def __repr__(self) -> str:
        repr = f"SFC_SET: {self.name}\n"
        for sfc in self.sfc_set:
            repr += f"{sfc}" + "\n"
        return repr
