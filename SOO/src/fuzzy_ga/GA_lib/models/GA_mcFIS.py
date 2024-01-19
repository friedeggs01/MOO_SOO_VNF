import numpy as np
from ..object.EA import Individual, Population
from ..operators import crossover, mutation, selection
from ..tasks.task import AbstractTask
import sys
from IPython.display import display, clear_output
import time
import random
import matplotlib.pyplot as plt
import os

class GA_mcFIS:
    def __init__(self, seed = 42, percent_print=0.5):
        self.history_cost: list[int] = []
        self.solve: list[Individual]
        self.seed = seed
        self.result = None
        self.generations = 100 # represent for 100% 
        self.display_time = True 
        self.clear_output = True 
        self.count_pre_line = 0
        self.printed_before_percent = -2
        self.percent_print = percent_print
    
    def compile(self,
        task: AbstractTask,
        crossover,
        mutation,
        selection
        ):
        self.task = task
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        set_seed(self.seed)

    def fit(self, nb_generations = 1000, nb_inds = 100, p_c = 0.8, p_m = 0.5, p_r=0.1, out_path=None) : # -> list[Individual]:
        self.time_begin = time.time()
        os.makedirs(out_path, exist_ok=True)

        # initial population
        self.population = Population(num_inds=nb_inds, task=self.task)
        self.history_cost.append([max(self.population.fitness)])
        self.render_process(0, ['Cost'], [self.history_cost[-1]], use_sys=True)

        # initial best individual
        self.global_best_individual = self.population.__getBestIndividiual__(max=True)
        self.arg_best = 0

        for epoch in range(nb_generations):
            offsprings = Population(num_inds=0, task=self.task)
            next_pop = Population(num_inds=0, task=self.task)

            # transfer p_r% of the best solution
            fitness = self.population.fitness
            num_transfered = max(1, int(len(fitness)*p_r))
            ids = np.argsort(np.array(fitness))[::-1][:num_transfered]
            for id in ids:
                next_pop.__addIndividual__(self.population.population[id])

            # crossover p_c%
            cnt_c = 0
            max_c = max(1, int(np.ceil(len(fitness) * p_c)))
            # print("Number of offsprings from crossover: ", max_c)
            while cnt_c < max_c:
                pa, pb = self.population.__getRWSIndividual__(size = 2)

                cnt = 0
                while((pa.genes - pb.genes).sum() == 0):
                    pa, pb = self.population.__getRWSIndividual__(size = 2)
                    if cnt > 10:
                        break
                    else: cnt += 1

                gen_a, gen_b = self.crossover(pa.genes, pb.genes)
                oa = Individual(gen_a, self.task)
                ob = Individual(gen_b, self.task)
                offsprings.__addIndividual__(oa)
                offsprings.__addIndividual__(ob)

                cnt_c += 2

            # mutation p_m%
            cnt_m = 0
            max_m = max(1, int(np.ceil(len(fitness) * p_m)))
            # print("Number of offsprings from mutation: ", max_m)
            while cnt_m < max_m:
                pa = self.population.__getRWSIndividual__(size=1)[0]
                gen_a = self.mutation(pa.genes)
                oa = Individual(gen_a, self.task)
                offsprings.__addIndividual__(oa)
 
                cnt_m += 1

            selected_idx = self.selection(offsprings, num_selected = nb_inds - num_transfered)
            for i in selected_idx:
                next_pop.__addIndividual__(offsprings.population[i])

            self.population = next_pop
            self.history_cost.append([max(self.population.fitness)])

            best_individual = self.population.__getBestIndividiual__(max=True)
            if best_individual.fitness > self.global_best_individual.fitness:
                self.global_best_individual = best_individual
                self.arg_best = epoch
                
            # log
            self.render_process((epoch+1)/nb_generations, ['Cost'], [self.history_cost[-1]], use_sys=True)
            with open(f"{out_path}/log.txt", "a") as f:
                f.write(f"Epoch: {epoch} | Best fitness: {best_individual.fitness} | Best gene: {list(best_individual.genes)} | List fitness: {best_individual.objective_list}\n")
            
            if epoch - self.arg_best > 40:
                break
                
        #print('END!')
        self.last_pop = self.population
        print(f"BEST SOLUTION: {self.global_best_individual.fitness}")
        with open(f"{out_path}/log.txt", "a") as f:
            f.write(f"FINAL BEST: {self.global_best_individual.fitness} | Best gene: {list(self.global_best_individual.genes)} | List fitness: {self.global_best_individual.objective_list}\n")


    def render_process(self,curr_progress, list_desc, list_value, use_sys = False, *args, **kwargs): 
        percent = int(curr_progress * 100)
        if percent >= 100: 
            self.time_end = time.time() 
            percent = 100 
        else: 
            if percent - self.printed_before_percent >= self.percent_print:
                self.printed_before_percent = percent 
            else: 
                return 
                
        process_line = '%3s %% [%-20s]  '.format() % (percent, '=' * int(percent / 5) + ">")
        
        seconds = time.time() - self.time_begin  
        minutes = seconds // 60 
        seconds = seconds - minutes * 60 
        print_line = str("")
        if self.clear_output is True: 
            if use_sys is True: 
                # os.system("cls")
                pass
            else:
                clear_output(wait= True) 
        if self.display_time is True: 
            if use_sys is True: 
                # sys.stdout.write("\r" + "time: %02dm %.02fs  "%(minutes, seconds))
                # sys.stdout.write(process_line+ " ")
                print_line = print_line + "Time: %02dm %2.02fs "%(minutes, seconds) + " " +process_line
                
            else: 
                display("Time: %02dm %2.02fs "%(minutes, seconds))
                display(process_line)
        for i in range(len(list_desc)):
            desc = str("")
            for value in range(len(list_value[i])):
                desc = desc + str("%d " % (list_value[i][value])) + " "
            line = '{}: {},  '.format(list_desc[i], desc)
            if use_sys is True: 
                print_line = print_line + line 
            else: 
                display(line)
        if use_sys is True: 
            # sys.stdout.write("\033[K")
            sys.stdout.flush() 
            sys.stdout.write("\r" + print_line)
            sys.stdout.flush() 

    def render_history(self, title = "input", yscale = None, ylim = None, re_fig = False, save_fig = True):
        nb_slot = [-cost[0] for cost in self.history_cost]
        plt.plot(np.arange(len(self.history_cost)), nb_slot)

        plt.title(title)
        plt.xlabel("Generations")
        plt.ylabel("Number of slot")
        
        if yscale is not None:
            plt.yscale(yscale)
        if ylim is not None:
            plt.ylim(bottom = ylim[0], top = ylim[1])
        if save_fig:
            plt.savefig(f"./image/{title}.png")    
        plt.show()

        # if re_fig:
        #     return fig


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)