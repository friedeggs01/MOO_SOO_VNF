import numpy as np
from .task import AbstractTask

class TimeTabling(AbstractTask):
    def __init__(self, data_path):
        self.data_path = data_path
        self.num_course: int
        self.num_room: int
        self.num_seat: np.ndarray
        self.num_candidate: np.ndarray
        self.conflict: list
        self.total_seat: int
        self.read_data()

    def read_data(self):
        with open(self.data_path, "r") as f:
            lines = f.readlines()
            self.num_course = int(lines[0])
            self.num_candidate = np.array(list(map(int, lines[1].split())))
            self.num_room = int(lines[2])
            self.num_seat = np.sort(np.array(list(map(int, lines[3].split()))))[::-1]
            self.total_seat = sum(self.num_seat)
            self.conflict = [[] for i in range(self.num_course)]
            K = int(lines[4])
            for i in range(K):
                course1, course2 = list(map(int,lines[5+i].split()))
                course1, course2 = course1 - 1, course2 - 1
                self.conflict[course1].append(course2)
                self.conflict[course2].append(course1)          

    # calculate fitness        
    def __call__(self, x):
        total_slot = np.max(x) + 1
        slot = [[] for i in range(self.num_course)]
        for course, s in enumerate(x):
            slot[s].append(course)
        #print(slot)
        # check feasible
        cnt_infeasbile = 0
        for slot_i in slot:
            if(len(slot_i) == 0):
                continue
            
            cnt_room = 0
            for i in range(len(slot_i)):
                for j in range(i+1,len(slot_i)):
                    if slot_i[j] in self.conflict[slot_i[i]]:
                        cnt_infeasbile += 1
                cur_num_candidate = self.num_candidate[slot_i[i]]
                #print(f"Course {slot_i[i]} has {cur_num_candidate} candidates - cnt_room: {cnt_room}")
                while(cur_num_candidate > 0):
                    if(cnt_room >= self.num_room):
                        cnt_infeasbile += 1
                        break
                    else:
                        cur_num_candidate -= self.num_seat[cnt_room]
                        cnt_room += 1  
        if(cnt_infeasbile):
            output = -(self.num_course + cnt_infeasbile)     
        else:
            output = -total_slot
        return output

    # initial randomly
    def encode(self):
        # genes = np.array([np.random.randint(self.num_course) for i in range(self.num_course)])
        # idx_sorted_genes = np.argsort(genes)
        # sorted_genes = np.empty_like(genes)
        # rank = 0
        # for idx, value in enumerate(idx_sorted_genes):
        #     sorted_genes[value] = rank
        #     if(idx > 0 and genes[idx_sorted_genes[idx-1]] < genes[idx_sorted_genes[idx]]):
        #         rank += 1
        #         sorted_genes[value] = rank
        # genes = sorted_genes
        # return genes
        return np.random.permutation(self.num_course)
