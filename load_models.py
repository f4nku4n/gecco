from search_spaces import SS_201

ss = SS_201()

def get_model(genotype, num_classes, taskcla):
    model = ss.get_model(genotype, num_classes=num_classes, taskcla=taskcla)
    return model
