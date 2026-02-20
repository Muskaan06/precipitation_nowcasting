# import numpy as np

# BASE_THRESHOLD = 0.5
# BETTER_VALUE = 2.5

# PERCENTAGE_THRESHOLD = 0.15
# BETTER_PERCENTAGE_THRESHOLD = 0.03

# BAD_THRESHOLD = 3
# BAD_MAX_THRESHOLD = 4
# HIGH_WEIGHTAGE = 5
# LOW_WEIGHTAGE = 2

# MINIMUM_PROBABILITY = 0.1




# # assuming input size = (B ,24, 224, 224)

# data = np.load("data/set20-30.npz", allow_pickle=True)
# dl = len(data['array'])
# print(dl)
# print(data['array'].shape)

# high = {}
# low = {}
# for i, sets in enumerate(data['array']):
#     t = []
#     c = 0
#     check = 0
#     print(sets.shape)
#     for image in sets:
#         print(image[0].shape)
#         #print(image[0])
#         a = image[0]>BASE_THRESHOLD
#         b = image[0]>BETTER_VALUE
#         print('a', np.sum(a)/(112*112))
#         print('b', np.sum(b)/(112*112))

#         if np.sum(b)/(112*112) >= BETTER_PERCENTAGE_THRESHOLD:
#             c = 0
#             t.append(2)
#         elif np.sum(a)/(112*112) >= PERCENTAGE_THRESHOLD:
#             c = 0
#             t.append(1)
#         else:
#             c+=1
#             t.append(0)
#             if c > BAD_THRESHOLD:
#                 check = 1
#                 if c > BAD_MAX_THRESHOLD:
#                     check = 2
#                     break
#     if not check:
#         high[i] = []
#         high[i].append(t)
#     elif check==1:
#         low[i] = []
#         low[i].append(t) 

# max_possible_score = 48

# for h in high:
#     score_sum = sum(high[h][0])
#     normalized_weight = (score_sum / max_possible_score) * HIGH_WEIGHTAGE
#     high[h].append(max(0.1, normalized_weight))

# for l in low:
#     score_sum = sum(low[l][0])
#     normalized_weight = (score_sum / max_possible_score) * LOW_WEIGHTAGE
#     low[l].append(max(0.1, normalized_weight))

# weights = [0]*dl

# for h in high:
#     weights[h] = high[h][1]
# for l in low:
#     weights[l] = low[l][1]

# print(weights)

# indices = list(range(len(weights)))

# def weighted_selection_once(indices, weights, num_to_select, min_prob=0.1):
#     indices = np.array(indices)
#     weights = np.array(weights, dtype=float)
    
#     total_weight = weights.sum()
#     if total_weight == 0:
#         return []
    
#     probs = weights / total_weight
#     print(probs)
    
#     valid_mask = probs >= min_prob
#     valid_indices = indices[valid_mask]
#     valid_probs = probs[valid_mask]
    
#     if len(valid_indices) == 0:
#         print("Warning: No sets met the minimum probability threshold.")
#         return []

#     valid_probs /= valid_probs.sum()

#     actual_num_to_select = min(num_to_select, len(valid_indices))
#     selected_indices = np.random.choice(
#         valid_indices, 
#         size=actual_num_to_select, 
#         replace=False, 
#         p=valid_probs
#     )
    
#     return selected_indices.astype(int).tolist()


# selected = weighted_selection_once(indices, weights, 8, min_prob = MINIMUM_PROBABILITY)
# print(f"Selected Set Indices: {selected}")

import numpy as np

def compute_weighted_selection(
    npz_dict,
    num_to_select,
    base_threshold=0.5,
    better_value=2.5,
    percentage_threshold=0.15,
    better_percentage_threshold=0.03,
    bad_threshold=3,
    bad_max_threshold=4,
    high_weightage=5,
    low_weightage=2,
    minimum_probability=0.1,
    max_possible_score=48
):
    print("-------------------------------------in random selection code!-----------------------------------")
    dl = len(npz_dict['array'])

    high = {}
    low = {}

    for i, sets in enumerate(npz_dict['array']):
        t = []
        c = 0
        check = 0

        for image in sets:
            a = image[0] > base_threshold
            b = image[0] > better_value

            #140 * 140 for himalayan region
            #112 *112 for north east and eastern region
            #85 * 85 for South

            if np.sum(b) / (112 * 112) >= better_percentage_threshold:
                c = 0
                t.append(2)
            elif np.sum(a) / (112 * 112) >= percentage_threshold:
                c = 0
                t.append(1)
            else:
                c += 1
                t.append(0)
                if c > bad_threshold:
                    check = 1
                    if c > bad_max_threshold:
                        check = 2
                        break

        if not check:
            high[i] = [t]
        elif check == 1:
            low[i] = [t]

    for h in high:
        score_sum = sum(high[h][0])
        normalized_weight = (score_sum / max_possible_score) * high_weightage
        high[h].append(normalized_weight)

    for l in low:
        score_sum = sum(low[l][0])
        normalized_weight = (score_sum / max_possible_score) * low_weightage
        low[l].append(normalized_weight)

    weights = [0] * dl
    for h in high:
        weights[h] = high[h][1]
    for l in low:
        weights[l] = low[l][1]

    indices = list(range(len(weights)))

    return weighted_selection_once(indices, weights, num_to_select, minimum_probability)


def weighted_selection_once(indices, weights, num_to_select, min_prob=0.1):

    indices = np.array(indices)
    weights = np.array(weights, dtype=float)

    total_weight = weights.sum()
    if total_weight == 0:
        return []

    probs = weights / total_weight
    valid_mask = probs >= min_prob

    valid_indices = indices[valid_mask]
    valid_probs = probs[valid_mask]

    if len(valid_indices) == 0:
        return []

    valid_probs /= valid_probs.sum()
    actual_num_to_select = min(num_to_select, len(valid_indices))

    selected_indices = np.random.choice(
        valid_indices,
        size=actual_num_to_select,
        replace=False,
        p=valid_probs
    )

    return selected_indices.astype(int).tolist()
