from random import choice

def max_index(vals):
    largest = max(vals)
    indices = [i for i, val in enumerate(vals) if val == largest]
    return choice(indices)