import numpy as np
from trecs.components import Creators

"""
Classes taken from algo_confounding
"""
class NewItemFactory(Creators):
    def __init__(self, items, items_per_iteration):
        self.items = items
        self.items_per_iteration = items_per_iteration
        self.idx = 0
    def generate_items(self):
        if self.idx > self.items.shape[1]:
            return np.empty((self.items.shape[0], 0))
            #raise RuntimeError("Ran out of items to generate!")
        idx_start = self.idx
        idx_end = self.idx + self.items_per_iteration
        self.idx = idx_end
        return self.items[:, idx_start:idx_end]
