import torch
from torch import Tensor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

BASE_THRESHOLD = 0.5
BETTER_VALUE = 2.5
PERCENTAGE_THRESHOLD = 0.5
BETTER_PERCENTAGE_THRESHOLD = 0.3
BAD_THRESHOLD = 3
BAD_MAX_THRESHOLD = 5



# assuming input size = (B ,24, 224, 224)

# add the npz opening code
np.random.seed(42)
dataset = np.random.rand(24,3, 224, 224)

dl = len(dataset)
print(dl)

high = {}
low = {}
for i, sets in enumerate(dataset):
    t = []
    c = 0
    check = 0
    for image in sets:
        a = image>BASE_THRESHOLD
        b = image>BETTER_VALUE

        if np.sum(b)/(224*224) >= BETTER_PERCENTAGE_THRESHOLD:
            c = 0
            t.append(2)
        elif np.sum(a)/(224*224) >= PERCENTAGE_THRESHOLD:
            c = 0
            t.append(1)
        else:
            c+=1
            t.append(0)
            if c > BAD_THRESHOLD:
                check = 1
                if c > BAD_MAX_THRESHOLD:
                    check = 2
                    break
    if not check:
        high[i] = []
        high[i].append(t)
    elif check==1:
        low[i] = []
        low[i].append(t)


#give weitghage to high and low 
HIGH_WEIGHTAGE = 1
LOW_WEIGHTAGE = 0.7

for h in high:
    a = sum(high[h][0])
    if a >= 25:
        high[h].append(1 * HIGH_WEIGHTAGE)
    elif a < 25 and a >= 15:
        high[h].append(0.7 * HIGH_WEIGHTAGE)
    elif a < 15 and a >=5:
        high[h].append(0.4* HIGH_WEIGHTAGE)
    else:
        high[h].append(0.1 * HIGH_WEIGHTAGE)

for l in low:
    a = sum(low[l][0])
    if a >= 25:
        low[l].append(1 * LOW_WEIGHTAGE)
    elif a < 25 and a >= 15:
        low[l].append(0.8 * LOW_WEIGHTAGE)
    elif a < 15 and a >=5:
        low[l].append(0.4* LOW_WEIGHTAGE)
    else:
        low[l].append(0.1 * LOW_WEIGHTAGE)
   
weights = [0]*dl

for h in high:
    weights[h] = high[h][1]
for l in low:
    weights[l] = low[l][1]

indices = list(range(len(weights)))

def weighted_selection_without_replacement(indices, weights, num_to_select):
    available_indices = np.array(indices)
    available_weights = np.array(weights, dtype=float)
    selected_sets = []

    for _ in range(num_to_select):
        if len(available_indices) == 0:
            break

        mask = available_weights > 0
        if not np.any(mask):
            break
            
        probs = available_weights[mask] / available_weights[mask].sum()
        

        chosen_idx_in_pool = np.random.choice(np.where(mask)[0], p=probs)
        
        selected_sets.append(int(available_indices[chosen_idx_in_pool]))
        
        available_indices = np.delete(available_indices, chosen_idx_in_pool)
        available_weights = np.delete(available_weights, chosen_idx_in_pool)
        
    return selected_sets

selected = weighted_selection_without_replacement(indices, weights, 3)
print(f"Selected Set Indices: {selected}")
