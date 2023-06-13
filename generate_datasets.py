from SpecMix.syntheticDatasetGeneration import generate_mixed_dataset
import numpy as np
import os

#Generate data

# sigmas = np.linspace(0, 4, 40)
# k_list = [2,3,4,5]
# p_list = [0.1,0.2,0.25,0.33,0.4]

# for k in k_list:
#     print(k)
#     for p in p_list:
#         for s in sigmas:
#             for i in range(1000):
#                 path = "Experiments/Synthetic Datasets/k=" + str(k) + "/p=" + str(p) +  "/sigma="+ str(s)
#                 if not os.path.exists(path):
#                     os.makedirs(path)
#                 filename = "Experiments/Synthetic Datasets/k=" + str(k) + "/p=" + str(p) +  "/sigma="+ str(s) + "/dataset " + str(i) + ".csv"
#                 generate_mixed_dataset(num_numerical_features=2, num_categorical_features=2, num_samples=1000, k=k, p=p, precomputed_centers=True, precomputed_sigma=s, save=True, filename=filename)

import multiprocessing
from functools import partial
sigmas = np.linspace(0, 4, 40)
k_list = [3,4,5]
p_list = [0.1,0.2,0.25,0.33,0.4]

def worker(i, s, p, k):
    path = f"Experiments/Synthetic Datasets/k={k}/p={p}/sigma={s}"
    os.makedirs(path, exist_ok=True)
    filename = f"{path}/dataset {i}.csv"
    generate_mixed_dataset(num_numerical_features=k, num_categorical_features=2, num_samples=1000, k=k, p=p, precomputed_centers=True, precomputed_sigma=s, save=True, filename=filename)

# get the number of CPUs available on the machine
num_cpu = multiprocessing.cpu_count()

# create a pool of worker processes
with multiprocessing.Pool(num_cpu) as pool:
    for k in k_list:
        print(k)
        for p in p_list:
            for s in sigmas:
                pool.map(partial(worker, s=s, p=p, k=k), range(1000))