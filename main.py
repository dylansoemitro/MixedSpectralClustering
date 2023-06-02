import numpy as np
import matplotlib.pyplot as plt
from SpecMix.benchmarks import calculate_score
from SpecMix.syntheticDatasetGeneration import generate_mixed_dataset
import time


def analyze_sigma(p, num_samples, num_numerical_features=2, num_categorical_features=1, num_experiments=1000, methods=['spectral'], metrics = ["jaccard", "purity"], knn=0, l=1):
    sigma_scores = {metric: {method: [] for method in methods} for metric in metrics}
    avg_time_taken = {method: [] for method in methods}
    if 'spectral' in methods:
        lambda_values = [0, 1, 100, 1000]
        lambda_keys = [f'spectral lambda={l}' for l in lambda_values]
        for k in lambda_keys:
            for metric in metrics:
                sigma_scores[metric][k] = []
            avg_time_taken[k] = []
    sigmas = np.linspace(0, 3, 20)

    for s in sigmas:
        exp_sigmas = {m: [] for m in methods}
        exp_time = {m: [] for m in methods}
        if 'spectral' in methods:
            for k in lambda_keys:
                exp_sigmas[k] = []
                exp_time[k] = []

        for _ in range(num_experiments):
            df = generate_mixed_dataset(num_numerical_features=num_numerical_features,
                                         num_categorical_features=num_categorical_features,
                                         num_samples=num_samples,
                                         p=p,
                                         precomputed_centers=True,
                                         precomputed_sigma=s)
            for m in methods:
                if m == 'spectral':
                    continue
                score, time_taken = calculate_score(df, df['target'].tolist(), 2, m, metrics=metrics, lambdas=[l]*num_categorical_features, knn=knn)
                exp_sigmas[m].append(score)
                exp_time[m].append(time_taken)
            if 'spectral' in methods:
                for l in lambda_values:
                    lambdas = [l] * num_categorical_features
                    score, time_taken = calculate_score(df, df['target'].tolist(), 2, 'spectral',  metrics=metrics, lambdas=lambdas, knn=knn)
                    exp_sigmas[f'spectral lambda={l}'].append(score)
                    exp_time[f'spectral lambda={l}'].append(time_taken)
        del exp_sigmas['spectral']
        del exp_time['spectral']
 
        for method, scores in exp_sigmas.items():
            for metric in metrics:
                mean_scores = []
                #print(scores)
                for dict in scores:
                    #print(dict)
                    mean_scores.append(dict[metric])
                mean_scores = np.mean(mean_scores)
                sigma_scores[metric][method].append(mean_scores)
            avg_time_taken[method].append(np.mean(exp_time[method]))

    for metric in metrics:
        del sigma_scores[metric]['spectral']
    del avg_time_taken['spectral']
    #print(sigma_scores)
    for metric in metrics:
        plt.figure()
        for method in sigma_scores[metric]:
            plt.plot(sigmas, sigma_scores[metric][method], label=method)

        # Add legend and labels to the plot
        plt.legend()
        plt.xlabel('Sigma')
        plt.ylabel(metric)
        plt.title(metric + 'Scores by Variance with Probability = ' + str(p))
        plt.savefig(metric + 'Scores_by_Variance_with_Probability=' + str(p) + '_plot.png')

        plt.show()

    for m in avg_time_taken:
        pass

    return sigma_scores


start_time = time.time()
scores = analyze_sigma(0.1, 1000, num_numerical_features = 2, num_categorical_features = 2, num_experiments=10, methods = ['spectral', 'k-prototypes', 'lca', 'spectralCAT'], l=1)
end_time = time.time()
print(scores)
print("Time taken:", end_time - start_time, "seconds")


start_time = time.time()
scores = analyze_sigma(0.4, 1000, num_numerical_features = 2, num_categorical_features = 2, num_experiments=10, methods = ['spectral', 'k-prototypes', 'lca', 'spectralCAT'], l=1)
end_time = time.time()
print(scores)
print("Time taken:", end_time - start_time, "seconds")

