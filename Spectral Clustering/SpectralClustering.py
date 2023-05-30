import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_bool_dtype
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph


def create_adjacency_df(df, lambdas=None, knn=0, return_df=False):
    """
    Creates an adjacency matrix for a given dataset for use in spectral clustering.

    Args:
    - df: pandas DataFrame, with shape (num_samples, num_features), the dataset
    - lambdas: list of ints, the distance between each pair of categorical variables
    - knn: int, the number of nearest neighbors to use for the KNN graph
    - return_df: boolean, whether to return a pandas DataFrame or a numpy array
    Returns:
    - matrix: numpy array/dataframe with shape (num_samples, num_samples), the adjacency matrix

    """

    # Initialize variables and data structures
    lambdas = lambdas or []
    numerical_nodes_count = len(df.index)
    df = df.drop(['target'], axis=1, errors='ignore')
    numerical_labels = []
    categorical_labels = []

    # Identify unique categorical variables
    unique_cols = []
    for col in df:
        if is_string_dtype(df[col]) or is_bool_dtype(df[col]):
            unique_cols.extend(df[col].unique())
    categorical_nodes_count = len(unique_cols)
    total_nodes_count = numerical_nodes_count + categorical_nodes_count

    # Initialize adjacency matrix
    matrix = np.zeros((total_nodes_count, total_nodes_count))

    # Separate numeric and categorical columns
    numeric_df = df.select_dtypes(include=np.number)
    categorical_df = df.select_dtypes(exclude=np.number)

    # Add numerical labels to list
    for i in range(numerical_nodes_count):
        numerical_labels.append(f'numerical{i}')

    # Calculate numerical distances using KNN graph or fully connected graph
    sigma = 1
    scaler = StandardScaler()
    numeric_df = np.array(numeric_df)

    if knn:
        A_dist = kneighbors_graph(numeric_df, n_neighbors=knn, mode='distance', include_self=True)
        A_conn = kneighbors_graph(numeric_df, n_neighbors=knn, mode='connectivity', include_self=True)
        A_dist = A_dist.toarray()
        A_conn = A_conn.toarray()

        # Make connectivity and distance matrices symmetric
        A_conn = 0.5 * (A_conn + A_conn.T)
        A_dist = 0.5 * (A_dist + A_dist.T)

        # Compute the similarities using boolean indexing
        dist_matrix = np.exp(-(A_dist)**2 / ((2 * sigma**2)))
        dist_matrix[~A_conn.astype(bool)] = 0.0
    else:
        dist_matrix = cdist(numeric_df, numeric_df, metric='euclidean')
        dist_matrix = np.exp(-(dist_matrix)**2 / ((2 * sigma**2)))

    # Add numerical distance matrix to the original one (top left corner)
    matrix[:numerical_nodes_count, :numerical_nodes_count] = dist_matrix

    # Add categorical labels to list
    for k, col in enumerate(categorical_df):
        for value in categorical_df[col].unique():
            categorical_labels.append(f'{col}={value}')

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

#def mixed_spectral_clustering(adj_matrix, n_clusters,  ):
