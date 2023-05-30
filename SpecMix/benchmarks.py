from sklearn.cluster import SpectralClustering
from sklearn.metrics import jaccard_score
from itertools import permutations
from kmodes.kprototypes import KPrototypes
from stepmix.stepmix import StepMix
from stepmix.utils import get_mixed_descriptor
from clustering import create_adjacency_df
from sklearn.metrics import confusion_matrix
import numpy as np
import time

def calculate_score(df, target_labels, n_clusters = 2, method = "spectral", metrics = ["jaccard"], lambdas=[], knn=0, binary_cols = [], categorical_cols = [], numerical_cols = []):
  df = df.drop(['target'], axis=1, errors='ignore')

  if method == "spectral":
    if lambdas:
        adj_matrix = create_adjacency_df(df, lambdas,knn=knn, numerical_cols = numerical_cols, categorical_cols = categorical_cols + binary_cols)
    else:
        adj_matrix = create_adjacency_df(df,knn=knn, numerical_cols = numerical_cols, categorical_cols = categorical_cols + binary_cols)
    start_time = time.time()
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans',random_state=0, affinity = 'precomputed').fit(adj_matrix)
    end_time = time.time()
    predicted_labels = clustering.labels_[:len(target_labels)].tolist()

  elif method == "k-prototypes":
    if categorical_cols:
        catColumnsPos = [df.columns.get_loc(col) for col in categorical_cols + binary_cols]
    else:
        catColumnsPos = [df.columns.get_loc(col) for col in list(df.select_dtypes('object').columns)]

    start_time = time.time()
    kprototypes = KPrototypes(n_jobs = -1, n_clusters = n_clusters, init = 'Huang', random_state = 0)
    predicted_labels = kprototypes.fit_predict(df.to_numpy(), categorical = catColumnsPos)
    end_time = time.time()

  elif method == "lca":
    # Extract binary columns
    binCols = set(df.select_dtypes(include=[bool]).columns)
    # Extract numerical columns
    numCols = set(df.select_dtypes(include=[int, float]).columns)
    # Extract categorical columns
    catCols = set(df.columns) - binCols - numCols

    start_time = time.time()
    mixed_data, mixed_descriptor = get_mixed_descriptor(
      dataframe=df,
      continuous=numCols,
      binary=binCols,
      categorical=catCols)
    model = StepMix(n_components=n_clusters, measurement=mixed_descriptor, verbose=1, random_state=123)

    # Fit model
    model.fit(mixed_data)

    # Class predictions
    df['mixed_pred'] = model.predict(mixed_data)
    predicted_labels = df['mixed_pred'].to_numpy()
    end_time = time.time()
  time_taken = end_time-start_time

  scores_dict = {}
  for metric in metrics:
    scores_list = []
    score_function = jaccard_score if metric == 'jaccard' else purity_score

    for perm in permutations(range(n_clusters)):
        perm_predicted_labels = [perm[label] for label in predicted_labels]
        score = score_function(perm_predicted_labels, target_labels)
        scores_list.append(score)

    max_score = max(scores_list)
    scores_dict[metric] = max_score

  return scores_dict, time_taken

def purity_score(y_pred, y_true):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Find the maximum value in each column (majority class)
    majority_sum = np.sum(np.amax(cm, axis=0))
    
    # Calculate purity
    purity = majority_sum / np.sum(cm)
    
    return purity