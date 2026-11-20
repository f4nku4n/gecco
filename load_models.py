import pickle as p
import numpy as np
from search_spaces import SS_201
from utils import count_parameters_in_MB, get_model_infos
from search_spaces.nb201.utils import dict2config, Structure, str2structure

ss = SS_201()

def get_model(genotype, num_classes, taskcla):
    model = ss.get_model(genotype, num_classes=num_classes, taskcla=taskcla)
    return model
