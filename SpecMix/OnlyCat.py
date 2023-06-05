
import time
import numpy as np
from scipy.sparse import spdiags, issparse
from scipy import sparse
from scipy.sparse.linalg import eigs, eigsh
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from syntheticDatasetGeneration import generate_mixed_dataset
from clustering import create_adjacency_df

np.set_printoptions(edgeitems=30, linewidth=100000)

def Tcut(B, Nseg):
    # B - |X|-by-|Y|, cross-affinity-matrix
    # note that |X| = |Y| + |I|
    B = sparse.csr_matrix(B)

    Nx, Ny = B.shape
    if Ny < Nseg:
        raise ValueError('Need more superpixels!')

    ### build the superpixel graph
    dx = B.sum(axis=1)
    Dx = spdiags(1 / dx.T, 0, Nx, Nx)
    Wy = B.T.dot(Dx).dot(B)

    ### compute Ncut eigenvectors
    # normalized affinity matrix
    d = Wy.sum(axis=1)
    D = spdiags(1.0 / np.sqrt(d.T), 0, Ny, Ny)
    nWy = D.dot(Wy).dot(D)
    nWy = (nWy + nWy.T) / 2

    # compute eigenvectors
    if issparse(nWy):
        evals_large, evecs_large = eigsh(nWy, k=Nseg, which='LM')
    else:
        evals_large, evecs_large = np.linalg.eigsh(nWy)
        evals_large = evals_large[::-1]
        evecs_large = evecs_large[:, ::-1]
    Ncut_evec = D.dot(evecs_large[:, :Nseg])

    ### compute the Ncut eigenvectors on the entire bipartite graph (transfer!)
    evec = Dx.dot(B).dot(Ncut_evec)

    ### k-means clustering
    # normalize each row to unit norm
    evec = evec / (np.sqrt(np.sum(evec ** 2, axis=1, keepdims=True)) + 1e-10)

    # k-means
    kmeans = KMeans(n_clusters=Nseg)
    labels = kmeans.fit_predict(evec)
    return labels

k = 3
n = 10000
p = 0.4

num_cf = 5
num_nf = 0
t = time.time()
df = generate_mixed_dataset(num_numerical_features=num_nf, num_categorical_features=num_cf, k=k, p=p, num_samples=n)
gt = df["target"].to_numpy()
print("DF: ", time.time() - t)

t = time.time()
A = create_adjacency_df(df)
print("A: ", time.time() - t)

B = A[0:n, n::]

t = time.time()
l = Tcut(B, k)
print("Tcut: ", time.time() - t)
print(adjusted_rand_score(gt, l))
print(l)