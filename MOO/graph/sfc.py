class SFC():
    def __init__(self, id, data_str=None):
        self.id = id
        self.source = int(data_str[3])
        self.destination = int(data_str[4])
        self.bw = int(data_str[0])
        self.memory = int(data_str[1])
        self.cpu = int(data_str[2])
        self.num_vnfs = int(data_str[5])
        self.vnf_list = list(data_str[6:])
        self.finished = False
        # the more duplicated vnf, the bigger density
        self.density = len(self.vnf_list) / len(set(self.vnf_list))
        self.path = [self.source]
        self.vnf_location = [] 
        
    def __len__(self):
        return len(self.vnf_list)