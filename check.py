from search_spaces import SS_201
import pickle as p
import numpy as np
from utils import count_parameters_in_MB, get_model_infos
from search_spaces.nb201.utils import dict2config, Structure, str2structure


if __name__ == '__main__':
    np.random.seed(0)
    ss = SS_201()
    list_phenotypes = []
    list_genotypes = []
    i = 0
    for x0 in range(5):
        for x1 in range(5):
            for x2 in range(5):
                for x3 in range(5):
                    for x4 in range(5):
                        for x5 in range(5):
                            genotype = [x0, x1, x2, x3, x4, x5]
                            phenotype = ss.decode(genotype)
                            list_genotypes.append(genotype)
                            list_phenotypes.append(str2structure(phenotype).to_unique_str(consider_zero=True))
    # print(list_phenotypes)
    I = np.unique(list_phenotypes, return_index=True)[1]
    list_unique_genotypes = np.array(list_genotypes)[I]
    p.dump(list_unique_genotypes, open('list_unique_genotypes.p', 'wb'))
    # phenotype = ss.sample()
    # genotype = ss.encode(phenotype)
    # network = ss.get_model(genotype)
    # print('Phenotype:', phenotype)
    # print('Genotype:', genotype)
    # # print('Network:', network)
    # print(count_parameters_in_MB(network))
    # print(get_model_infos(network, (1, 3, 32, 32)))
    # # pass

# import json

# data = json.load(open('databases/zc_nasbench201.json', 'r'))
# print(len(data['cifar10']))