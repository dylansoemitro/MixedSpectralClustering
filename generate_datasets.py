from SpecMix.syntheticDatasetGeneration import generate_mixed_dataset


#Generate data
for i in range(1000):
    filename = "datasets/mixed_dataset_" + str(i) + ".csv"
    generate_mixed_dataset(num_numerical_features=2, num_categorical_features=2, num_samples=1000, k=2, p=0.1, precomputed_centers=True, precomputed_sigma=0.2, save=True, filename=filename)