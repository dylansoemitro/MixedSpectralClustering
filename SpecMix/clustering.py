import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_bool_dtype
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

def create_adjacency_df(df, sigma = 1, automatic_sigma=False, lambdas=None, knn=0, return_df=False, numerical_cols=[], categorical_cols=[], n_clusters = 2):
    """
    Creates an adjacency matrix for a given dataset for use in spectral clustering.

    Args:
    - df: pandas DataFrame, with shape (num_samples, num_features), the dataset
    - sigma: float, the sigma value for the Gaussian kernel
    - lambdas: list of ints, the distance between each pair of categorical variables
    - knn: int, the number of nearest neighbors to use for the KNN graph
    - return_df: boolean, whether to return a pandas DataFrame or a numpy array
    - numerical_cols: list of strings, the names of the numerical columns
    - categorical_cols: list of strings, the names of the categorical columns
    Returns:
    - matrix: numpy array/dataframe with shape (num_samples, num_samples), the adjacency matrix
    """

    # Initialize variables and data structures
    lambdas = lambdas or []
    numerical_nodes_count = len(df.index)
    df = df.drop(['target'], axis=1, errors='ignore')

    numerical_labels = []  # keep track of numerical node labels
    categorical_labels = []  # keep track of categorical node labels

    # If columns are not specified, use the original logic
    if numerical_cols is None and categorical_cols is None:
        # Separate numeric and categorical columns
        numeric_df = df.select_dtypes(include=np.number)
        categorical_df = df.select_dtypes(exclude=np.number)
    else:
        numeric_df = df[numerical_cols]
        categorical_df = df[categorical_cols]

    # Add numerical labels to list
    for i in range(numerical_nodes_count):
        numerical_labels.append(f'numerical{i}')

    # Add categorical labels to list
    for k, col in enumerate(categorical_df):
        for value in categorical_df[col].unique():
            categorical_labels.append(f'{col}={value}')
    # Identify unique categorical variables
    unique_cols = []
    for col in df:
        if is_string_dtype(df[col]) or is_bool_dtype(df[col]):
            unique_cols.extend(df[col].unique())
    categorical_nodes_count = len(unique_cols)
    total_nodes_count = numerical_nodes_count + categorical_nodes_count

    # Initialize adjacency matrix
    matrix = np.zeros((total_nodes_count, total_nodes_count))

    # Add numerical labels to list
    for i in range(numerical_nodes_count):
        numerical_labels.append(f'numerical{i}')

    # Calculate numerical distances using KNN graph or fully connected graph
    sigma = 1
    if automatic_sigma:
        if automatic_sigma == "median_pairwise":
            sigma = median_pairwise(numeric_df)
        elif automatic_sigma == "ascmsd":
            sigma = ascmsd(numeric_df, knn)
        elif "cv_distortion":
            sigmas = np.linspace(0.1, 100, 1000)
            sigma = cv_distortion_sigma(numeric_df, sigmas, n_clusters=n_clusters, lambdas = lambdas, knn=knn, categorical_cols=categorical_cols, numerical_cols=numerical_cols)
        elif "cv_sigma":
            sigmas = np.linspace(0.1, 100, 1000)
            sigma = cv_sigma(numeric_df, sigmas, n_clusters=n_clusters, knn=knn)
        else:
            raise ValueError("Invalid automatic_sigma value. Must be one of: median_pairwise, ascmsd, cv_distortion, cv_sigma")
    scaler = StandardScaler()
    numeric_arr = scaler.fit_transform(np.array(numeric_df))

    if knn:
        A_dist = kneighbors_graph(numeric_arr, n_neighbors=knn, mode='distance', include_self=True)
        A_conn = kneighbors_graph(numeric_arr, n_neighbors=knn, mode='connectivity', include_self=True)
        A_dist = A_dist.toarray()
        A_conn = A_conn.toarray()

        # Make connectivity and distance matrices symmetric
        A_conn = 0.5 * (A_conn + A_conn.T)
        A_dist = 0.5 * (A_dist + A_dist.T)

        # Compute the similarities using boolean indexing
        dist_matrix = np.exp(-(A_dist)**2 / ((2 * sigma**2)))
        dist_matrix[~A_conn.astype(bool)] = 0.0
    else:
        dist_matrix = cdist(numeric_arr, numeric_arr, metric='euclidean')
        dist_matrix = np.exp(-(dist_matrix)**2 / ((2 * sigma**2)))

    # Add numerical distance matrix to the original one (top left corner)
    matrix[:numerical_nodes_count, :numerical_nodes_count] = dist_matrix

    # Connect categorical nodes to numerical observations
    for i in range(numerical_nodes_count):
        for k, col in enumerate(categorical_df):
            j = numerical_nodes_count + categorical_labels.index(f'{col}={categorical_df[col][i]}')
            if not lambdas:
                matrix[i][j], matrix[j][i] = 1, 1
            else:
                matrix[i][j], matrix[j][i] = lambdas[k], lambdas[k]

    # Create labeled DataFrame if required
    if return_df:
        return pd.DataFrame(matrix, index=numerical_labels + categorical_labels, columns=numerical_labels + categorical_labels)
    else:
        return matrix


def median_pairwise(numeric_arr):
    # Compute pairwise distances
    pairwise_distances = kneighbors_graph(numeric_arr)
    sigma = np.median(pairwise_distances)
    return sigma

def ascmsd(numeric_arr, knn):
    # Compute pairwise distances
    pairwise_distances = kneighbors_graph(numeric_arr, n_neighbors=knn, mode='distance')
    pairwise_distances = pairwise_distances.toarray()

    # Estimate density
    density = np.exp(-pairwise_distances**2 / 2.0)
    density = np.sum(density, axis=1)

    # Compute density peaks
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(numeric_arr)
    density_peaks = neigh.kneighbors(numeric_arr[density.argsort()[::-1]])[1][:, 1]

    # Compute pairwise distances between density peaks
    density_peak_distances = pairwise_distances[density_peaks][:, density_peaks]

    # Estimate scale
    sigma = np.median(density_peak_distances)
    return sigma

def cv_distortion_sigma(df, sigmas, n_clusters, lambdas=[], knn=0, numeric_columns=None, categorical_columns=None):
    min_distortion = float('inf')
    best_sigma = None

    for sigma in sigmas:
        # Create the adjacency matrix for the current sigma value
        adjacency_matrix = create_adjacency_df(df, sigma, lambdas, knn, False, numeric_columns, categorical_columns)

        # Perform spectral clustering
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=0)
        labels = clustering.fit_predict(adjacency_matrix)

        # Calculate the distortion of the clusters
        feature_matrix = normalize(clustering.affinity_matrix_, axis=1)
        distortion = 0
        for i in range(n_clusters):
            cluster_points = feature_matrix[labels == i, :]
            cluster_center = cluster_points.mean(axis=0)
            squared_distances = np.sum((cluster_points - cluster_center) ** 2, axis=1)
            distortion += np.sum(squared_distances)

        # Update the best sigma if the current distortion is lower
        if distortion < min_distortion:
            min_distortion = distortion
            best_sigma = sigma

    return best_sigma


def cv_sigma(adjacency_matrix, sigma_values, n_clusters, scoring_function=silhouette_score):
    """
    This function computes the best sigma via cross-validation.
    sigma_values is a list of sigma values to try.
    n_clusters is the number of clusters to use for Spectral Clustering.
    """
    best_sigma = None
    best_score = -np.inf

    for sigma in sigma_values:
        # Apply spectral clustering with the current sigma
        sc = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=1.0/sigma**2)
        cluster_labels = sc.fit_predict(adjacency_matrix)

        # Compute score
        score = scoring_function(adjacency_matrix, cluster_labels)

        # Update the best score and best sigma if current score is higher
        if score > best_score:
            best_score = score
            best_sigma = sigma

    return best_sigma
