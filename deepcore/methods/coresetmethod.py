class CoresetMethod(object):
    def __init__(self, loader, configs, args, robust_learner, fraction=0.5, random_seed=None, **kwargs):
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Illegal Coreset Size.")
        self.configs = configs
        self.loader = loader
        self.robust_learner = robust_learner
        self.num_classes = args.n_class #len(dst_train.classes)
        self.fraction = fraction
        self.random_seed = random_seed
        self.index = []
        self.args = args

        self.n_train = loader.n_train
        self.coreset_size = round(self.n_train * fraction)

    def select(self, **kwargs):
        return

