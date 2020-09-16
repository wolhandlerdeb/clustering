# author: deborah wolhandler

import numpy as np
from scipy.stats import randint,norm,multivariate_normal, ortho_group
from scipy.linalg import subspace_angles, orth
from scipy import optimize
from scipy import linalg
from sklearn.cluster import KMeans
from cluster.selfrepresentation import ElasticNetSubspaceClustering
import seaborn as sns
import pandas as pd
from itertools import permutations
import math
from sklearn.decomposition import PCA


'''1. Simulation Study, Noiseless case. Simulate data according to the model above with the following parameters:
- n = 2^3,2^4,...,2^10.
- p = 2^4,2^5,2^6,2^ d 7.
- d = 2^(−1)p, 2^(−2)p, 2^(−3)p,2^(−4)p, for each of the values of p.
- K = 4 clusters.
- θ = 10^(−2)θmax, 10^(−1)θmax, θmax, where θmax is the value obtained on average by taking
the different subspaces Bi to have uniformly random orientations. (when p ≫ d, 1 we
have θmax ≈ π but for small p, θmax can be significantly smaller)

Remark 2. To simulate the subspaces B1,..,BK,
you can simulate for each subspace K unit vectors in random directions forming a basis.
This yields an average pairwise angle denote θmax which depends on the dimensions p and d.
For high dimension p ≫ d random subspaces are almost orthogonal and θmax ≈ π .


To simulate subspaces with a given pre-specified average largest principal angle θ < θmax,
you can first simulate subspaces B1, .., BK in random directions, and an additional random shared subspace B0.
Then, replace each Bi by a linear combination Bi ← αBi + (1 − α)B0 where α ∈ (0, 1) is calibrated to yield average
pairwise angle θ.
'''''


# zi ∼U({1,..,K})
# wi ∼N(0,Id)
# xi|zi,wi ∼N(Bziwi,σ2Ip)

# def generate_rand_data(B, k, n,p,dim, sigma):
#     z = randint.rvs(1,k, size=n)
#     w = multivariate_normal.rvs(
#         mean=np.zeros(dim), cov=np.identity(dim),
#                                  size=n)
#     B = np.array(B)# todo to check
#     # todo to check : B[:,z,:]*w
#     x = multivariate_normal.rvs(mean=B[:,z,:]*w, cov=(sigma ** 2) * np.identity(p),
#                                size=n)  # todo to check
#
#     return z,w,x
def generate_rand_data(B, k, n,p,dim, sigma=1):
    z = np.random.randint(0, k, n)
    w = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma*np.diag(np.ones(dim)), size=n)
    X = np.zeros((n, p))
    for i in range(n):
        X[i,] = np.random.multivariate_normal(mean=np.array(np.dot(np.array(w[i, :]), B[z[i]].T)).flatten(),
                                              cov=np.diag(np.ones(p)))  # sigma value is missing
    return z,w,X


def run_algo(K,dataX ):
    pass


def run_():
    pass

def pca_subspace(x, i, dim):
    pca_components_number = 3
    pca = PCA(n_components=pca_components_number)
    pca.fit_transform(x)
    B_kmeans = pca.components_
    return B_kmeans.T

def sim_orth_basis(p, dim, k):
    b = [orth(np.random.rand(p, dim)) for i in range(k + 1)]
    return b


def find_theta_max(b, t, k):
    theta_max = []
    for i in range(1, k + 1):
        for j in range(1, i):
            theta_max.append(subspace_angles(b[i], b[j]).max())
    max_avg_theta = np.mean(theta_max)
    theta = max_avg_theta * t
    return theta


#second_simulation
def fixed_orth_basis(b,k,theta):
    def find_alpha_for_theta(a, b=b, k=k, theta=theta):
        temp_theta = []
        for i in range(1, k + 1):
            for j in range(1, i):
                temp_theta.append(subspace_angles(b[0] * (1 - a) + b[i] * a, b[0] * (1 - a) + b[j] * a).max())
        return np.mean(temp_theta) - theta

    a = optimize.bisect(find_alpha_for_theta, 0, 1)
    B = [b[0] * (1 - a) + b[i] * a for i in range(1, k + 1)]
    return B


# Recovery Performance
def measure_cost_subspace(k, B1, B2):
    all_perm = list(permutations(range(k)))
    sum_cos_angles_all_per = np.zeros(len(all_perm))
    for l, perm in enumerate(all_perm):
        for i in range(k):
            if B2[perm[i]].shape[1] > 0:  # handling with empty clusters
                sum_cos_angles_all_per[l] += math.cos(
                    subspace_angles(B1[i], B2[perm[i]]).max()) ** 2  # use min or max????????????????
    cost_subspace = sum_cos_angles_all_per.max()
    return cost_subspace
    # WHAT ARE WE DOING WITH EMPTY CLUSTERS


def measure_cost_cluster(k, cluster1, cluster2):
    data = {'cluster1': cluster1, 'cluster2': cluster2}
    clusters = pd.DataFrame(data, index=range(len(cluster1)))
    all_perm = list(permutations(range(k)))
    accuracy_rate_all_per = np.zeros(len(all_perm))
    for l, perm in enumerate(all_perm):
        c = [i for i in range(k)]
        dic = dict(zip(c, perm))
        clusters['premut_cluster'] = clusters['cluster2'].transform(lambda x: dic[x] if x in dic else None)
        m = clusters.groupby(['cluster1', 'premut_cluster']).size().unstack(fill_value=0)
        accuracy_rate_all_per[l] = np.trace(m)
    cost_cluster = accuracy_rate_all_per.max() / len(cluster1)
    return cost_cluster

def print_heatmap():
    pass

def q_a():
    k = 4
    n_vals = [2 ** j for j in range(3, 11)]
    p_vals = [2 ** j for j in range(4, 8)]
    d_vals = [2 ** -j for j in range(1, 5)]
    t_vals= [10 ** -j for j in range(0, 3)]
    sigma = 0  # noiseless
    kmeans = KMeans(n_clusters=k)
    for n in n_vals:
        for p in p_vals:
            for d in d_vals:
                dim = int(d * p)
                b =sim_orth_basis(p=p ,dim=dim ,k=k)
                for t in t_vals:
                    theta = find_theta_max(b=b, t=t, k=k)
                    B = fixed_orth_basis(b,k,theta=theta)
                    z, w ,x = generate_rand_data(B,k,n,p,dim,sigma)

                    kmeans_fit =kmeans.fit(x)
                    B_kmean = [pca_subspace(x, i, dim) for i in range(k)]
                    measure_cost_cluster(k, B, x['B_kmean'])
                    measure_cost_subspace(k, B, x['B_kmean'])
                    # sns.heatmap()
                    print(kmeans_fit.labels_)
                    # todo check parameter gamma
                    model_ensc = ElasticNetSubspaceClustering(n_clusters=k, algorithm='spams', gamma=500)
                    model_ensc.fit()
                    print(model_ensc.labels_)


def q_b():
    pass


def main():
    q_a()

if __name__ == '__main__':
    main()



