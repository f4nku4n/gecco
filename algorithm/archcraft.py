import random
import time

from evo_utils import StatusUpdateTool, Utils, Log
from genetic.population import Population, Individual
from genetic.evaluate import FitnessEvaluate
from genetic.mutation import Mutation
import numpy as np
import copy

from genetic import Individual

class EvolveCNN:
    """
    ArchCraft - Revisiting Neural Networks for Continual Learning: An Architectural Perspective (IJCAI 2024)
    """
    def __init__(self, popSize, maxGen, problem, evaluator):
        self.parent_pops = None
        self.pop = []
        self.hashPop = []
        self.pop_size = popSize
        self.maxGen = maxGen
        self.problem = problem  # Search space
        self.evaluator = evaluator  # Fitness evaluator

    def initialize(self):
        for _ in self.pop_size:
            idv = Individual()
            idv.X = self.problem.search_space.sample()
            idv.hashX = ''.join([str(x) for x in idv.X])
            if idv.hashX in self.hashPop:
                continue
            self.hashPop.append(idv.hashX)
            idv.phenotype = self.problem.search_space.decode(idv.X)
            idv.model = self.problem.search_space.get_model(idv.X)
            self.pop.append(idv)

    def evaluate(self, X):
        for idv in X:
            fitness_score = self.evaluator(idv)
            idv.F = fitness_score

    def run(self):
        self.initialize()
        self.evaluate(self.pop)
        for curr_gen in range(gen_no, max_gen):
            self.generate_offspring()
            self.evaluate()
            self.environment_selection()
        self.generate_offspring()

    def generate_offspring(self):
        cm = Mutation(Log, self.pops.individuals, _params={'gen_no': self.pops.gen_no})
        offspring = cm.process(mut_offspring_num=self.params['pop_size'])

        _str = []
        for ind in offspring:
            _str.append(str(ind))
            _str.append('-' * 100)
        file_name = './populations/offspring_%02d.txt' % self.pops.gen_no
        with open(file_name, 'w') as f:
            f.write('\n'.join(_str))

        self.parent_pops = copy.deepcopy(self.pops)
        self.pops.individuals = copy.deepcopy(offspring)

    def environment_selection(self):
        indi_list = []
        for indi in self.pops.individuals:
            indi_list.append(indi)
        for indi in self.parent_pops.individuals:
            indi_list.append(indi)
        pop_size = self.params['pop_size']
        elitism = 0.2
        e_count = int(pop_size * elitism)
        indi_list.sort(key=lambda x: x.acc, reverse=True)
        # descending order
        next_individuals = indi_list[0:e_count]
        left_list = indi_list[e_count:]
        np.random.shuffle(left_list)

        for i in range(pop_size - e_count):
            idx1 = random.randrange(0, len(left_list))
            indi1 = left_list.pop(idx1)
            idx2 = random.randrange(0, len(left_list))
            indi2 = left_list.pop(idx2)
            if indi1.acc > indi2.acc:
                next_individuals.append(indi1)
            else:
                next_individuals.append(indi2)

        """Here, the population information should be updated, such as the gene no and then to the individual id"""
        next_gen_pops = Population(self.pops.params, self.pops.gen_no)
        next_gen_pops.create_from_offspring(next_individuals)
        self.pops = next_gen_pops

        Utils.save_population_at_begin(str(self.pops), self.pops.gen_no)
        self.pops.gen_no += 1
