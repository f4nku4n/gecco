import random

from genetic.population import Population, Individual
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
        self.popSize = popSize
        self.maxGen = maxGen
        self.problem = problem  # Search space
        self.evaluator = evaluator  # Fitness evaluator

    def initialize(self):
        for _ in self.popSize:
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
        for nGen in range(self.maxGen):
            O = self.generate_offspring()
            self.evaluate(O)
            PO = self.pop + O
            self.selection(PO)
        self.generate_offspring()

    def generate_offspring(self):
        cm = Mutation(Log, self.pops.individuals, _params={'gen_no': self.pops.gen_no})
        offspring = cm.process(mut_offspring_num=self.params['pop_size'])

        O = []
        return O

    def selection(self, PO):
        indi_list = []
        elitism = 0.2
        e_count = int(self.popSize * elitism)
        indi_list.sort(key=lambda x: x.acc, reverse=True)
        # descending order
        next_individuals = indi_list[0:e_count]
        left_list = indi_list[e_count:]
        np.random.shuffle(left_list)

        for i in range(self.popSize - e_count):
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
