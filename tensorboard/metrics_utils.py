class MetricAccuracyClassification(object):
    def __init__(self):
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.tn = 0

    def __add__(self, other):
        self.fp += other.fp
        self.fn += other.fn
        self.tp += other.tp
        self.tn += other.tn
        return self

    def __iadd__(self, other):
        self.__add__(other)
        return self

    def __len__(self):
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self):
        if (self.tp + self.tn) == 0:
            if self.fp == 0 and self.fn == 0:
                return 1.0
            return 0
        return (self.tp + self.tn) / len(self)

    @property
    def recall(self):
        if (self.tp + self.tn) == 0:
            if self.fn == 0:
                return 1.0
            return 0
        return (self.tp + self.tn) / ((self.tp + self.tn) + self.fn)

    @property
    def precision(self):
        if (self.tp + self.tn) == 0:
            if self.fp == 0:
                return 1.0
            return 0
        return (self.tp + self.tn) / ((self.tp + self.tn) + self.fp)

    @property
    def f1(self):
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
