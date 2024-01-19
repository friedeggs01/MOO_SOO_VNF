import copy
class SFC():
    def __init__(self, id, data_str=None):
        self.id = id
        self.source = int(data_str[3])
        self.destination = int(data_str[4])
        self.bw = float(data_str[0])
        self.memory = float(data_str[1])
        self.cpu = float(data_str[2])
        self.num_vnfs = int(data_str[5])
        self.vnf_list = list(map(lambda x: int(x), data_str[6:]))
        self.finished = False
        self.density = len(self.vnf_list) / len(set(self.vnf_list))

        # addition for version 2
        self.finished = False
        self.path = [self.source]
        self.vnf_location = []
        self.vnf_demand = copy.copy(self.vnf_list)

    def __len__(self):
        return len(self.vnf_list)

    def __repr__(self, ) -> str:
        try:
            return "SFC {}: {} -> {} | VNF_require: {} | Finished: {} | VNF location: {} | Path: {}".format(self.id, self.source, self.destination, self.vnf_list, self.finished, self.vnf_location, self.path)
        except:
            return "SFC {}".format(self.id)

    def __str__(self) -> str:
        return self.__repr__()