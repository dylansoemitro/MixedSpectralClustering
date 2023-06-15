import pandas as pd
from SpecMix.benchmarks import calculate_score
import os
import pickle as pkl


def real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename):
    scores = {}
    avg_time_taken = {}
    lambda_values = [0, 1, 100, 1000]
    # if 'spectral' in methods:
    #     lambda_values = [0, 10, 100, 1000]
    #     lambda_keys = []
    #     if kernel:
    #         for ker in kernel:
    #             lambda_keys.extend([f'spectral lambda={l} kernel={ker}' for l in lambda_values])               
    #     else:
    #         lambda_keys = [f'spectral lambda={l}' for l in lambda_values]
    #     # for k in lambda_keys:
    #     #     for metric in metrics:
    #     #         scores[metric][k] = 0
    #     #     avg_time_taken[k] = 0
    # # del scores['spectral']
    # # del avg_time_taken['spectral']

    # Load the CSV into a DataFrame
    df = pd.read_csv(path + filename, names=column_names)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


    # Replace '?' with NaN
    df.replace('?', pd.NA, inplace=True)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    df['target'] = df['target'].astype('category').cat.codes

    for m in methods:
        if m == 'spectral':
            continue
        score, time_taken, _ = calculate_score(df, df['target'].tolist(), num_clusters, m, metrics=metrics, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
        scores[m] = score
        avg_time_taken[m] = time_taken
    if 'spectral' in methods:
        for ker in kernel:
            curr_kernel = 0
            for l in lambda_values:
                if kernel:
                    if not curr_kernel:
                        score, time_taken, curr_kernel = calculate_score(df, df['target'].tolist(), num_clusters, 'spectral',  metrics=metrics, lambdas=[l] * len(categorical_cols), numerical_cols=numerical_cols, categorical_cols=categorical_cols, kernel=ker)
                    else:
                        score, time_taken, curr_kernel = calculate_score(df, df['target'].tolist(), num_clusters, 'spectral',  metrics=metrics, lambdas=[l] * len(categorical_cols), numerical_cols=numerical_cols, categorical_cols=categorical_cols, kernel=None, curr_kernel=curr_kernel)
                    scores[f'spectral lambda={l} kernel={ker}'] = score
                    avg_time_taken[f'spectral lambda={l} kernel={ker}'] = time_taken
                else:
                    lambdas = [l] * len(categorical_cols)
                    score, time_taken, _ = calculate_score(df, df['target'].tolist(), num_clusters, 'spectral',  metrics=metrics, lambdas=lambdas, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
                    scores[f'spectral lambda={l}'] = score
                    avg_time_taken[f'spectral lambda={l}'] = time_taken
            curr_kernel = 0
        
    #Save scores and time taken
    #make directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/scores.pkl', 'wb') as f:
        pkl.dump(scores, f)
    with open(f'{path}/avg_time_taken.pkl', 'wb') as f:
        pkl.dump(avg_time_taken, f)
    return scores, avg_time_taken

methods = ['spectral', 'k-prototypes', 'lca', 'spectralCAT', 'denseclus']
metrics = ['jaccard', 'purity', 'calinski_harabasz', 'silhouette', 'adjusted_rand', 'homogeneity']
num_clusters = 3
kernel = ['median_pairwise', 'cv_sigma', 'preset']
path = 'Experiments/Real Datasets/Iris/'
filename = 'iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
categorical_cols = []
numerical_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename)
print(scores)

#Wine

# path = 'Experiments/Real Datasets/Wine/'
# filename = 'wine.data'
# column_names  = ["target", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "date", "proline"]
# numerical_cols = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "date", "proline"]
# scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename)
# print(scores)

