import pandas as pd
import numpy as np


def generate_mixed_dataset(num_numerical_features=2, num_categorical_features=1, num_samples=10, k=2, mu_range=(-1,1), sigma_range = (0.1,0.3), p=0, precomputed_centers = False, precomputed_sigma = 0.2):
    """
    Generates a mixed type dataset with numerical and categorical features and k clusters.

    Args:
    - num_numerical_features: int, the number of numerical features in the dataset
    - num_categorical_features: int, the number of categorical features in the dataset
    - num_samples: int, the number of samples in the dataset
    - k: int, the number of clusters to generate
    - mu_range: tuple (float, float), the range of values for the mean of the numerical features
    - sigma_range: tuple (float, float), the range of values for the standard deviation of the numerical features
    - p: float, the noise value for the categorical features
    - precomputed_centers: boolean for 2 clusters, means of -1, 1.
    Returns:
    - df: pandas DataFrame with shape (num_samples, num_numerical_features + num_categorical_features + 1), the features and the target label of the dataset
    """

    # Situation where we want to generate 2 clusters with precomputed centers
    if precomputed_centers:
      mu = np.array([[1,0], [-1,0]])
      sigma = np.array([[precomputed_sigma, precomputed_sigma],
                       [precomputed_sigma, precomputed_sigma]])
      k=2
    #Generate the numerical features based on Gaussian distributions
    else:
      mu = np.random.uniform(mu_range[0], mu_range[1], size=(k, num_numerical_features))
      sigma = np.random.uniform(sigma_range[0], sigma_range[1], size=(k, num_numerical_features))
    cluster_idx = np.random.randint(k, size=num_samples)
    cluster_idx = np.sort(cluster_idx)
    X_num = np.zeros((num_samples, num_numerical_features))
    for i in range(k):
        idx = np.where(cluster_idx == i)[0]
        X_num[idx] = np.random.normal(mu[i], sigma[i], size=(len(idx), num_numerical_features))

    # Generate the categorical features
    categories = []
    for i in range(k):
        cluster_categories = np.random.choice(10, size=num_categorical_features, replace=False)
        categories.append(cluster_categories)
        
    X_cat = np.zeros((num_samples, num_categorical_features))
    X_cat = np.tile(cluster_idx, (num_categorical_features,1))

    mask = np.random.choice([True, False], size=(num_categorical_features, num_samples), p=[p, 1-p])

    # randomly replace elements in the matrix with new values
    new_values = np.random.randint(0, k, size=(num_categorical_features, num_samples))

    for i in range(num_categorical_features):
      for j in range(num_samples):
        if mask[i][j] == True: 
          if X_cat[i][j] == new_values[i][j]:
            X_cat[i][j] = (new_values[i][j]+1) % k
          else:
            X_cat[i][j] = new_values[i][j]

    
    # Generate random names for categorical features
    cat_names = [f"cat_feat_{i}" for i in range(num_categorical_features)]

    # Combine the numerical and categorical features
    df = pd.DataFrame(np.concatenate([X_num, X_cat.T], axis=1), columns=[f"num_feat_{i}" for i in range(num_numerical_features)] + cat_names)
    cat_feat_cols = [col for col in df.columns if 'cat_feat' in col]
    df[cat_feat_cols] = df[cat_feat_cols].astype(str) 

    # Add the target label
    df['target'] = cluster_idx


    return df