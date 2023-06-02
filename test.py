import numpy as np
from sklearn.metrics import jaccard_score
from itertools import permutations
import time
from scipy.optimize import linear_sum_assignment
import ot
from collections import Counter

def compute_max_jaccard(labels1, labels2):
    # Calculate all possible permutations
    permuts = permutations(labels1)

    max_jaccard = 0
    for p in permuts:
        # Compute jaccard score
        jaccard = jaccard_score(np.array(p), np.array(labels2), average='micro')
        # Update maximum jaccard score
        if jaccard > max_jaccard:
            max_jaccard = jaccard

    return max_jaccard

import numpy as np
import ot
from collections import Counter

import numpy as np
import ot
from collections import Counter

def compute_optimized_jaccard(labels1, labels2):
    # Count the frequency of each label
    counter1 = Counter(labels1)
    counter2 = Counter(labels2)

    # Create distributions
    size = max(max(counter1.keys()), max(counter2.keys())) + 1
    distr1 = np.zeros(size)
    distr2 = np.zeros(size)
    
    for key, value in counter1.items():
        distr1[key] = value
    for key, value in counter2.items():
        distr2[key] = value

    # Normalize distributions
    distr1 = (distr1 + 1e-8) / np.sum(distr1 + 1e-8)  # Add a small value to avoid division by zero
    distr2 = (distr2 + 1e-8) / np.sum(distr2 + 1e-8)  # Add a small value to avoid division by zero

    # Compute cost matrix
    cost_matrix = np.ones((size, size)) - np.eye(size)

    # Compute the Sinkhorn distance
    sinkhorn_dist = ot.sinkhorn(distr1, distr2, cost_matrix, 0.1)

    # Compute the jaccard score
    jaccard_score = 1 - np.sum(sinkhorn_dist)

    return jaccard_score

labels1 = [0, 3, 2, 3, 7, 7]
labels2 = [5, 0, 3, 2, 1, 0]

print(compute_optimized_jaccard(labels1, labels2))


labels1 = [0, 3, 2, 3, 7, 7]
labels2 = [5, 0, 3, 2, 1, 0]
start_time = time.time()
print(compute_max_jaccard(labels1, labels2))
end_time = time.time()
print("Time taken:", end_time - start_time, "seconds")

start_time = time.time()
print(compute_optimized_jaccard(labels1, labels2))
end_time = time.time()
print("Time taken:", end_time - start_time, "seconds")