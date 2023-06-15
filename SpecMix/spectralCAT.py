from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import calinski_harabasz_score
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time
import pandas as pd

def categorize(X, max_k=10):
    S = {}
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, n_init='auto')
        labels = kmeans.fit_predict(X)
        S[k] = calinski_harabasz_score(X, labels)

    # Finding kbest
    local_max_k_values = [k for k in range(3, max_k) if S[k-1] < S[k] > S[k+1]]
    kbest = min(local_max_k_values) if local_max_k_values else max_k

    # Applying kmeans clustering with kbest
    kmeans = KMeans(n_clusters=kbest)
    X_cat = kmeans.fit_predict(X)

    return X_cat

#Used implementation from https://github.com/ztacs/Clustering-High-dimensional-Noisy-Categorical-Data/blob/main/SpectralCAT/spectralCAT.m
def spectralCAT(df, r, replicate, ran):
    np.random.seed(ran)
    m, n = df.shape
    n = n - 1  # removing the last column (ground truth clustering label)
    
    cat = df.nunique().values  # number of categories of each attribute

    for l in range(n):
      if pd.api.types.is_numeric_dtype(df.iloc[:, l]):
          df[df.columns[l]] = categorize(df.iloc[:, l].values.reshape(-1,1))
    cat = df.nunique().values  # number of categories of each attribute
    label_encoder = LabelEncoder()
    df = df.apply(lambda x: label_encoder.fit_transform(x) if x.dtype == 'object' else x)


    # Pairwise distance matrix Delta
    valid_entries = ~(np.isnan(df) | (df == 999))
    Delta = np.zeros((m, m))
    for k in range(n):
        data_k = df.iloc[:, k].values[:, None]  # Add new axis to apply broadcasting
        valid_k = valid_entries.iloc[:, k].values

        # Compute a pairwise difference matrix for current attribute, and divide by the number of categories
        diff_matrix = np.abs(data_k - data_k.T) / cat[k]

        # Update Delta only for valid pairwise combinations
        Delta += np.where(valid_k & valid_k.T, diff_matrix, 0)
    Delta += Delta.T - np.diag(Delta.diagonal())
    
    # Define X_bar in eq(4)
    percent = np.percentile(Delta, 100/3, axis=1)
    X_bar = np.zeros(m)
    c = np.zeros(m, dtype=int)
    for i in range(m):
        mask = Delta[i, :] <= percent[i]
        X_bar[i] = np.mean(Delta[i, mask])
        c[i] = np.sum(mask)
        
    # Define epsilon by hamming distance
    epsilon = np.zeros(m)
    for i in range(m):
        mask = Delta[i, :] <= percent[i]
        epsilon[i] = np.mean((Delta[i, mask] - X_bar[i])**2)

    # Eqn(3) part of eqn (5)
    omeg = np.zeros(m)
    for i in range(m):
        if X_bar[i] != 0:
            mask = Delta[i, :] <= percent[i]
            omeg[i] = np.sum(np.exp(-Delta[i, mask] / epsilon[i]))
        else:
            omeg[i] = c[i]

    # Adaptive Gaussian kernel eqn(6)
    W = np.exp(-Delta / np.sqrt(np.outer(omeg, omeg)))
    # d = np.sum(W, axis=1)
    # P = W / np.sqrt(np.outer(d, d))

    # U, S, _ = svds(P, r)
    # Pr_P = U.dot(np.diag(S))

    # # Perform k-means clustering
    # start_time = time.time()
    # #print("n_clusters: ", r)
    # kmeans = KMeans(n_clusters=r, n_init=replicate, max_iter=100, random_state=ran)
    # C = kmeans.fit_predict(Pr_P)

    #Spectral Clustering

    start_time = time.time()
    clustering = SpectralClustering(n_clusters=r, assign_labels = 'kmeans', affinity = 'precomputed').fit(W)
    C = clustering.labels_

    T = df.iloc[:, n].values
    Acc = accuracy_score(T, C)
    et = time.time() - start_time

    return Acc, et, C, T
