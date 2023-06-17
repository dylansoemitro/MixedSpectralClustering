from sklearn.cluster import SpectralClustering
from sklearn.metrics import jaccard_score, silhouette_score, calinski_harabasz_score, adjusted_rand_score, homogeneity_score, confusion_matrix
from itertools import permutations
from kmodes.kprototypes import KPrototypes
from stepmix.stepmix import StepMix
from stepmix.utils import get_mixed_descriptor
from SpecMix.clustering import create_adjacency_df
from SpecMix.spectralCAT import spectralCAT
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from denseclus import DenseClus
from prince import FAMD
import numpy as np
import time
from collections import Counter
import pandas as pd
import gower
def calculate_score(df, target_labels, n_clusters = 2, method = "spectral", metrics = ["jaccard"], lambdas=[], knn=0, binary_cols = [], categorical_cols = [], numerical_cols = [], kernel=None, curr_kernel=0):
  sigma = 0
  
  print("Method: ", method)
  if method == "spectral":
    df = df.drop(['target'], axis=1, errors='ignore')
    if lambdas:
        adj_matrix, sigma = create_adjacency_df(df, lambdas=lambdas,knn=knn, numerical_cols = numerical_cols, categorical_cols = categorical_cols + binary_cols, kernel=kernel, sigma=curr_kernel)
    else:
        adj_matrix, sigma = create_adjacency_df(df,knn=knn, numerical_cols = numerical_cols, categorical_cols = categorical_cols + binary_cols,  kernel=kernel, sigma=curr_kernel)
    start_time = time.time()
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans',random_state=0, affinity = 'precomputed').fit(adj_matrix)
    end_time = time.time()
    predicted_labels = clustering.labels_[:len(target_labels)].tolist()

  elif method == "k-prototypes":
    df = df.drop(['target'], axis=1, errors='ignore')
    if categorical_cols:
        catColumnsPos = [df.columns.get_loc(col) for col in categorical_cols + binary_cols]
    else:
        catColumnsPos = [df.columns.get_loc(col) for col in list(df.select_dtypes('object').columns)]

    if catColumnsPos and not len(numerical_cols) == df.shape[1]:
      print("k-protypes")
      kprototypes = KPrototypes(n_jobs = -1, n_clusters = n_clusters, init = 'Huang', random_state = 0)
      start_time = time.time()
      predicted_labels = kprototypes.fit_predict(df.to_numpy(), categorical = catColumnsPos)
      end_time = time.time()

    else:
      #do kmeans
      start_time = time.time()
      kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df.to_numpy())
      end_time = time.time()
      predicted_labels = kmeans.labels_

  elif method == "lca":
    df = df.drop(['target'], axis=1, errors='ignore')
    # Extract binary columns

    if not categorical_cols:
      # Extract numerical columns
      numCols= set(df.select_dtypes(include=[int, float]).columns)
      # Extract categorical columns
      binCols = set(df.select_dtypes(include=[bool]).columns)

      catCols = set(df.columns) - numCols - binCols
    else:
      catCols = set(categorical_cols)
      numCols = set(numerical_cols)
      binCols = set(binary_cols)
    if not catCols and not binary_cols:
      model = StepMix(n_components=n_clusters, measurement="continuous", verbose=1, random_state=123)
      mixed_data = df
    else:
      mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=df,
        continuous=numCols,
        binary=binCols,
        categorical=catCols)
      model = StepMix(n_components=n_clusters, measurement=mixed_descriptor, verbose=0, random_state=123)
    # Fit model
    start_time = time.time()
    model.fit(mixed_data)
    end_time = time.time()
    # Class predictions
    df['mixed_pred'] = model.predict(mixed_data)
    predicted_labels = df['mixed_pred'].to_numpy()
    end_time = time.time()
  elif method == "spectralCAT":
    start_time = time.time()
    _, _, predicted_labels, _= spectralCAT(df.copy(), n_clusters, 30, 0)
    end_time = time.time()
  elif method == "famd":
    df = df.drop(['target'], axis=1, errors='ignore')
    famd = FAMD(n_components=n_clusters, random_state=0, copy=True)
    start_time = time.time()
    famd.fit(df)
    transformed = famd.transform(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(transformed)
    predicted_labels = kmeans.predict(transformed)
    end_time = time.time()

  elif method == "denseclus":
    df = df.drop(['target'], axis=1, errors='ignore')
    print(df.dtypes)
    denseclus = DenseClus(umap_combine_method="intersection_union_mapper", cluster_selection_method="leaf", random_state=0)
    start_time = time.time()
    denseclus.fit(df)
    end_time = time.time()
    predicted_labels = denseclus.score()

  else:
    raise ValueError("Invalid method")

  time_taken = end_time-start_time
  scores_dict = {}
  # print("predicted labels: ", predicted_labels)
  # print("target labels: ", target_labels)
  element_frequencies_pred = Counter(predicted_labels)
  element_frequencies_target = Counter(target_labels)
  print("Predicted Label Frequencies: ")
  for element, frequency in element_frequencies_pred.items():
      print(f"Element {element}: {frequency} times")
  print("Target Label Frequencies: ")
  for element, frequency in element_frequencies_target.items():
      print(f"Element {element}: {frequency} times")
  score_function_dict = {"jaccard": jaccard_score, "purity": purity_score, "silhouette": silhouette_score, "calinski_harabasz": calinski_harabasz_score, "adjusted_rand": adjusted_rand_score, "homogeneity": homogeneity_score}
  for metric in metrics:
    scores_list = []
    score_function = score_function_dict[metric]
    if score_function == jaccard_score:
      for perm in permutations(range(n_clusters)):
          perm_predicted_labels = [perm[label] for label in predicted_labels]
          if n_clusters > 2:
            score = score_function(perm_predicted_labels, target_labels, average='weighted')
          else:
            score = score_function(perm_predicted_labels, target_labels)
          scores_list.append(score)
      score = max(scores_list)
    elif score_function == silhouette_score or score_function == calinski_harabasz_score:

      # if categorical_cols:
      #   catColumns = df[categorical_cols + binary_cols]
      # else:
      #   catColumns = df.select_dtypes(include=[object, bool])
      #   numerical_cols = df.select_dtypes(include=[int, float])
      df = df.drop(['target'], axis=1, errors='ignore')
      gower_dist_matrix = gower.gower_matrix(df)      

      #one hot encode all categorical columns
      # if catColumns.shape[1] > 0:
      #   encoder = OneHotEncoder()
      #   df_encoded = encoder.fit_transform(catColumns)
      #   df_encoded  = pd.concat([df[numerical_cols], pd.DataFrame(df_encoded.toarray())], axis=1)

      # if df_encoded.shape[0] == 0:
      #   print("here")
      #   encoder = OneHotEncoder()
      #   df_encoded = encoder.fit_transform(catColumns)
      #   df_encoded  = pd.concat([df[numerical_cols], pd.DataFrame(df_encoded.toarray())], axis=1)

      if len(np.unique(predicted_labels)) == 1:
        score = -1
      else:
        if score_function == silhouette_score:
          score = score_function(gower_dist_matrix, predicted_labels, metric = 'precomputed')
        else:
          score = score_function(gower_dist_matrix, predicted_labels)
    else:
      score = score_function(predicted_labels, target_labels)
    scores_dict[metric] = score
  return scores_dict, time_taken, sigma

def purity_score(y_pred, y_true):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Find the maximum value in each column (majority class)
    majority_sum = np.sum(np.amax(cm, axis=0))
    
    # Calculate purity
    purity = majority_sum / np.sum(cm)
    
    return purity