import numpy as np
import matplotlib.pyplot as plt
from SpecMix.benchmarks import calculate_score
from SpecMix.syntheticDatasetGeneration import generate_mixed_dataset
import time
import pandas as pd
import argparse
import os

import warnings

# # Suppress all warnings
warnings.filterwarnings("ignore")

def analyze_sigma(p, num_samples, num_numerical_features=2, num_categorical_features=1, num_experiments=1000, methods=['spectral'], metrics = ["jaccard", "purity"], knn=0, l=1, num_clusters=2, kernel=[], generated=False, directory=None):
    sigma_scores = {metric: {method: [] for method in methods} for metric in metrics}
    avg_time_taken = {method: [] for method in methods}
    if 'spectral' in methods:
        lambda_values = [0, 10, 100, 1000]
        lambda_keys = []
        if kernel:
            for ker in kernel:
                lambda_keys.extend([f'spectral lambda={l} kernel={ker}' for l in lambda_values])               
        else:
            lambda_keys = [f'spectral lambda={l}' for l in lambda_values]
        for k in lambda_keys:
            for metric in metrics:
                sigma_scores[metric][k] = []
            avg_time_taken[k] = []
    sigmas = np.linspace(0, 4, 40)
    #remove every other sigma
    sigmas = sigmas[::2]
    numerical_cols = [f'num_feat_{i}' for i in range(num_numerical_features)]
    categorical_cols = [f'cat_feat_{i}' for i in range(num_categorical_features)]
    for s in sigmas:
        print("sigma:", s)
        start = time.time()
        exp_sigmas = {m: [] for m in methods}
        exp_time = {m: [] for m in methods}
        if 'spectral' in methods:
            for k in lambda_keys:
                exp_sigmas[k] = []
                exp_time[k] = []

        for i in range(num_experiments):
            if not generated:
                df = generate_mixed_dataset(num_numerical_features=num_numerical_features,
                                            num_categorical_features=num_categorical_features,
                                            num_samples=num_samples,
                                            p=p,
                                                k=num_clusters,
                                            precomputed_centers=True,
                                            precomputed_sigma=s)
            else:
                filename = directory + "/k=" + str(num_clusters) + "/p=" + str(p) +  "/sigma="+ str(s) + "/dataset " + str(i) + ".csv"
                df = pd.read_csv(filename)
                
            for m in methods:
                if m == 'spectral':
                    continue
                score, time_taken, _ = calculate_score(df, df['target'].tolist(), num_clusters, m, metrics=metrics, lambdas=[l]*len(categorical_cols), knn=knn, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
                exp_sigmas[m].append(score)
                exp_time[m].append(time_taken)
            if 'spectral' in methods:
                for ker in kernel:
                    curr_kernel = 0

                    for l in lambda_values:
                        if kernel:
                            if not curr_kernel:
                                score, time_taken, curr_kernel = calculate_score(df, df['target'].tolist(), num_clusters, 'spectral',  metrics=metrics, lambdas=[l] * len(categorical_cols), knn=knn, numerical_cols=numerical_cols, categorical_cols=categorical_cols, kernel=ker)
                            else:
                                score, time_taken, curr_kernel = calculate_score(df, df['target'].tolist(), num_clusters, 'spectral',  metrics=metrics, lambdas=[l] * len(categorical_cols), knn=knn, numerical_cols=numerical_cols, categorical_cols=categorical_cols, kernel=None, curr_kernel=curr_kernel)
                            
                            exp_sigmas[f'spectral lambda={l} kernel={ker}'].append(score)
                            exp_time[f'spectral lambda={l} kernel={ker}'].append(time_taken)
                        else:
                            lambdas = [l] * len(categorical_cols)
                            score, time_taken, _ = calculate_score(df, df['target'].tolist(), num_clusters, 'spectral',  metrics=metrics, lambdas=lambdas, knn=knn, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
                            exp_sigmas[f'spectral lambda={l}'].append(score)
                            exp_time[f'spectral lambda={l}'].append(time_taken)
                    curr_kernel = 0
        del exp_sigmas['spectral']
        del exp_time['spectral']
 
        for method, scores in exp_sigmas.items():
            for metric in metrics:
                mean_scores = []
                for dict in scores:
                    mean_scores.append(dict[metric])
                mean_scores = np.mean(mean_scores)
                sigma_scores[metric][method].append(mean_scores)
            avg_time_taken[method].append(np.mean(exp_time[method]))
        end = time.time()
        print("Time taken:", end - start)
    for metric in metrics:
        del sigma_scores[metric]['spectral']
    del avg_time_taken['spectral']
    path = directory + '/Results/p=' + str(p) + '/k=' + str(num_clusters) + '/'
    #make directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    for ker in kernel:
        for metric in metrics:
            plt.figure()
            for method in sigma_scores[metric]:
                if ker in method:
                    plt.plot(sigmas, sigma_scores[metric][method], label=method)
                elif 'spectral' not in method or method == 'spectralCAT':
                    plt.plot(sigmas, sigma_scores[metric][method], label=method)

            # Add legend and labels to the plot
            plt.legend()
            plt.xlabel('Sigma')
            plt.ylabel(metric)
            plt.title(metric + ' Scores by Variance with Probability = ' + str(p))
            plt.savefig(path + metric + '_kernel=' + ker+'_plot.png')
            plt.show()

    plt.figure()
    for ker in kernel:
        for m in avg_time_taken:
            if ker in m:
                plt.plot(sigmas, avg_time_taken[m], label=m)
            elif 'spectral' not in m:
                plt.plot(sigmas, avg_time_taken[m], label=m)
    

        # Add legend and labels to the plot
        plt.legend()
        plt.xlabel('Sigma')
        plt.ylabel('Time taken')
        plt.title('Time taken by Variance with Probability = ' + str(p))
        plt.savefig(path + 'kernel=' + ker + '_time_plot.png')

        plt.show()

    #Save the scores and time taken to a csv file
    df = pd.DataFrame()
    df['sigma'] = sigmas
    for metric in metrics:
        for method in sigma_scores[metric]:
            df[method + '_' + metric] = sigma_scores[metric][method]
    for method in avg_time_taken:
        df[method + '_time'] = avg_time_taken[method]
    df = df.reset_index(drop=True)
    df.to_csv(path + 'scores.csv')
    return sigma_scores


def plot(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.plot(x, y)

    # Add legend and labels to the plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)

    plt.show()





# if __name__ == "__main__":
#     #command line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--p", type=float, default=0.1, help="Probability of a categorical feature")
#     parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
#     parser.add_argument("--num_numerical_features", type=int, default=2, help="Number of numerical features")
#     parser.add_argument("--num_categorical_features", type=int, default=2, help="Number of categorical features")
#     parser.add_argument("--num_experiments", type=int, default=1000, help="Number of experiments")
#     parser.add_argument("--methods", nargs="+", default=["spectral"], help="Methods to use")
#     parser.add_argument("--metrics", nargs="+", default=["jaccard"], help="Metrics to use")
#     parser.add_argument("--knn", type=int, default=0, help="Number of nearest neighbors to use for the KNN graph")
#     parser.add_argument("--l", type=int, default=1, help="Distance between each pair of categorical variables")
#     parser.add_argument("--num_clusters", type=int, default=2, help="Number of clusters")
#     parser.add_argument("--kernel", type=str, default="", help="Method to determine to use for spectral clustering")
#     parser.add_argument("--generate", type=bool, default=False, help="Generate the datasets")
#     parser.add_argument("--directory", type=str, default="datasets", help="Directory to save the datasets")
#     args = parser.parse_args()
#     start_time = time.time()
#     scores = analyze_sigma(args.p, args.num_samples, args.num_numerical_features, args.num_categorical_features, args.num_experiments, args.methods, args.metrics, args.knn, args.l, args.num_clusters, args.kernel, args.generate, args.directory)
#     end_time = time.time()
#     print("Time taken:", end_time - start_time, "seconds")

# start_time = time.time()
# scores = analyze_sigma(0.4, 1000, num_numerical_features = 2, num_categorical_features = 2, num_experiments=10, methods = ['spectral', 'k-prototypes', 'lca', 'spectralCAT'], l=1)
# end_time = time.time()
# print("Time taken:", end_time - start_time, "seconds")


p = 0.4
num_samples = 1000
num_numerical_features = 2
num_categorical_features = 2
num_experiments = 1
methods = ['spectral', 'k-prototypes', 'lca', 'spectralCAT', 'denseclus']
metrics = ['purity', 'calinski_harabasz', 'homogeneity', 'silhouette', 'adjusted_rand']
num_clusters = 2
kernel = ['median_pairwise', 'cv_sigma', 'preset']
generate =  False
directory = 'Experiments/Synthetic Datasets'
analyze_sigma(0.1, num_samples, num_numerical_features=num_numerical_features, num_categorical_features=num_categorical_features, num_experiments=num_experiments, methods=methods, metrics=metrics, num_clusters=num_clusters, kernel=kernel, generated=generate, directory=directory)
analyze_sigma(0.25, num_samples, num_numerical_features=num_numerical_features, num_categorical_features=num_categorical_features, num_experiments=num_experiments, methods=methods, metrics=metrics, num_clusters=num_clusters, kernel=kernel, generated=generate, directory=directory)
analyze_sigma(p, num_samples, num_numerical_features=num_numerical_features, num_categorical_features=num_categorical_features, num_experiments=num_experiments, methods=methods, metrics=metrics, num_clusters=num_clusters, kernel=kernel, generated=generate, directory=directory)
analyze_sigma(0.1, num_samples, num_numerical_features=num_numerical_features, num_categorical_features=num_categorical_features, num_experiments=num_experiments, methods=methods, metrics=metrics, num_clusters=3, kernel=kernel, generated=generate, directory=directory)
analyze_sigma(0.25, num_samples, num_numerical_features=num_numerical_features, num_categorical_features=num_categorical_features, num_experiments=num_experiments, methods=methods, metrics=metrics, num_clusters=3, kernel=kernel, generated=generate, directory=directory)
analyze_sigma(p, num_samples, num_numerical_features=num_numerical_features, num_categorical_features=num_categorical_features, num_experiments=num_experiments, methods=methods, metrics=metrics, num_clusters=3, kernel=kernel, generated=generate, directory=directory)
analyze_sigma(0.1, num_samples, num_numerical_features=num_numerical_features, num_categorical_features=num_categorical_features, num_experiments=num_experiments, methods=methods, metrics=metrics, num_clusters=4, kernel=kernel, generated=generate, directory=directory)
analyze_sigma(0.25, num_samples, num_numerical_features=num_numerical_features, num_categorical_features=num_categorical_features, num_experiments=num_experiments, methods=methods, metrics=metrics, num_clusters=4, kernel=kernel, generated=generate, directory=directory)
analyze_sigma(p, num_samples, num_numerical_features=num_numerical_features, num_categorical_features=num_categorical_features, num_experiments=num_experiments, methods=methods, metrics=metrics, num_clusters=4, kernel=kernel, generated=generate, directory=directory)
