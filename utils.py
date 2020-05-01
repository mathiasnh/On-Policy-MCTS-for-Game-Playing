from random import choice
import numpy as np

def max_index(vals):
    largest = max(vals)
    indices = [i for i, val in enumerate(vals) if val == largest]
    return choice(indices)

def rescale(state, distribution):
    length = len(state) - 1
    r = np.zeros(length)
    for i in range(length):
        if state[i] == 0:
            r[i] = distribution.pop(0)
    return r

def normalize(a):
    return a / np.linalg.norm(a, ord=1)