import numpy as np
from ..object.EA import Individual, Population
from ..operators import crossover, mutation, selection
from ..tasks.task import AbstractTask
import sys
from IPython.display import display, clear_output
import time
import random
import matplotlib.pyplot as plt

class GA_model:
    def __init__(self, seed = None, percent_print=0.5):
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
        self.crossover = crossover(task.num_course)
        self.mutation = mutation(task.num_course)
        self.selection = selection()

    def fit(self, nb_generations = 1000, nb_inds = 100, p_c = 0.8, p_m = 0.5) -> list[Individual]:
        self.time_begin = time.time()
        # initial population
        population = Population(num_inds=nb_inds, task=self.task)
        #print(f"First generation:{population.__getGenes__()}")
        self.history_cost.append([max(population.fitness)])
        self.render_process(0, ['Cost'], [self.history_cost[-1]], use_sys=True)
        for epoch in range(nb_generations):
            offsprings = Population(num_inds=0, task=self.task)
            #print(f"EPOCH {epoch} - current best fitness: {self.history_cost[-1]}")
            while len(offsprings) < len(population):
                p1, p2 = population.__getRandomIndividual__(size = 2)
                p3, p4 = population.__getRandomIndividual__(size = 2)
                if(p1.fitness > p2.fitness):
                    pa = p1
                else:
                    pa = p2
                if(p3.fitness > p4.fitness):
                    pb = p3
                else:
                    pb = p4
                if np.random.rand() < p_c:
                    gen_a, gen_b = self.crossover(pa.genes, pb.genes)
                    oa = Individual(gen_a, self.task)
                    ob = Individual(gen_b, self.task)
                    offsprings.__addIndividual__(oa)
                    offsprings.__addIndividual__(ob)
                if np.random.rand() < p_m:
                    gen_a = self.mutation(pa.genes)
                    gen_b = self.mutation(pb.genes)
                    oa = Individual(gen_a, self.task)
                    ob = Individual(gen_b, self.task)
                    #print(f"Genes {pa.genes} & {pb.genes} ({max(pa.fitness, pb.fitness)})-> {gen_a} & {gen_b} ({max(oa.fitness,ob.fitness)}")
                    offsprings.__addIndividual__(oa)
                    offsprings.__addIndividual__(ob)
            
            for cnt, ind in enumerate(offsprings.population):
                if cnt >= nb_inds:
                    break
                population.__addIndividual__(ind)
            selected_idx = self.selection(population)
            new_pop = Population(num_inds=0, task=self.task)
            for i in selected_idx:
                new_pop.__addIndividual__(population.population[i])
            population = new_pop
            self.history_cost.append([max(population.fitness)])

            # log
            #print(f'Epoch {epoch} Best fitness: {self.history_cost[-1]}')
            self.render_process((epoch+1)/nb_generations, ['Cost'], [self.history_cost[-1]], use_sys=True)
            #print(f'len population:{len(population)} - len new_pop:{len(new_pop)}')
        #print('END!')
        self.last_pop = population
        #print(f"The last generation {population.__getGenes__()}")
        print(f"BEST SOLUTION: {self.history_cost[-1]}")
        return self.last_pop

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

    def render_history(self, title = "input", yscale = None, ylim: list[float, float] = None, re_fig = False, save_fig = True):
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


