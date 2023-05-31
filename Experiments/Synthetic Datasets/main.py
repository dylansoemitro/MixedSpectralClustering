import numpy as np
import matplotlib.pyplot as plt
from SpecMix.benchmarks import calculate_score
from SpecMix.syntheticDatasetGeneration import generate_mixed_dataset

def analyze_sigma(p, num_samples, num_numerical_features=2, num_categorical_features=1, num_experiments=1000, methods=['spectral'], knn=0, l=1):
    sigma_scores = {m: [] for m in methods}
    avg_time_taken = {m: [] for m in methods}
    if 'spectral' in methods:
        lambda_values = [0, 1, 120, 2000]
        lambda_keys = [f'spectral lambda={l}' for l in lambda_values]
        for k in lambda_keys:
            sigma_scores[k] = []
            avg_time_taken[k] = []
        print(lambda_keys)
    sigmas = np.linspace(0, 3, 20)

    for s in sigmas:
        exp_sigmas = {m: [] for m in methods}
        if 'spectral' in methods:
            for k in lambda_keys:
                exp_sigmas[k] = []

        for _ in range(num_experiments):
            df = generate_mixed_dataset(num_numerical_features=num_numerical_features,
                                         num_categorical_features=num_categorical_features,
                                         num_samples=num_samples,
                                         p=p,
                                         precomputed_centers=True,
                                         precomputed_sigma=s)
            for m in methods:
                score, time_taken = calculate_score(df, df['target'].tolist(), 2, m, lambdas=[l]*num_categorical_features, knn=knn)
                exp_sigmas[m].append(score)
                avg_time_taken[m].append(time_taken)

            if 'spectral' in methods:
                for l in lambda_values:
                    lambdas = [l] * num_categorical_features
                    score, time_taken = calculate_score(df, df['target'].tolist(), 2, 'spectral', lambdas=lambdas, knn=knn)
                    exp_sigmas[f'spectral lambda={l}'].append(score)
                    avg_time_taken[f'spectral lambda={l}'].append(time_taken)

        for method, scores in exp_sigmas.items():
            mean_score = np.mean(scores)
            sigma_scores[method].append(mean_score)

    for m in sigma_scores:
        plt.plot(sigmas, sigma_scores[m], label=m)

    # Add legend and labels to the plot
    plt.legend()
    plt.xlabel('Sigma')
    plt.ylabel('Jaccard Score')
    plt.title('Jaccard Scores by Variance with Probability = ' + str(p))
    plt.show()

    for m in avg_time_taken:
        mean_val = np.mean(avg_time_taken[m])
        print(f"Average time taken for {m}: {mean_val}")

    return sigma_scores


start_time = time.time()
scores = analyze_sigma(0.1, 1000, num_numerical_features = 2, num_categorical_features = 2, num_experiments=100, methods = ['spectral', 'k-prototypes', 'lca'], l=1)
end_time = time.time()
print(scores)
print("Time taken:", end_time - start_time, "seconds")

#plot the highest average (area under curve)
#one above
#one below
#one in between for both
