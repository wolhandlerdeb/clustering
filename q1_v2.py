import numpy as np
import pandas as pd
import scipy as sc
from scipy.stats import randint, norm, multivariate_normal, ortho_group
from scipy import linalg
from scipy.linalg import subspace_angles, orth
from scipy.optimize import fmin
from scipy import optimize
from scipy.optimize import minimize
import math
from statistics import mean, stdev, variance
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import itertools as it
import matplotlib.pyplot as plt
import sys
from cluster.selfrepresentation import ElasticNetSubspaceClustering
from scipy.optimize import linear_sum_assignment


def first_simulation(p, dim, k):
    b = [orth(np.random.randn(p, dim)) for i in range(k + 1)]
    return (b)


# This yields an average pairwise angle denote θmax which depends on the dimensions p and d
def find_theta_max(p, dim):
    theta_max = []
    for i in range(100):
        rand_subspac1 = orth(np.random.randn(p, dim))
        rand_subspac2 = orth(np.random.randn(p, dim))
        theta_max.append(subspace_angles(rand_subspac1, rand_subspac2).max())  # using min or max
    max_avg_theta = np.average(theta_max)
    return (max_avg_theta)


# Then, replace each Bi by a linear combination Bi ← αBi + (1 − α)B0 where α ∈ (0, 1) is calibrated to yield average pairwise angle θ
def second_simulation(p, k, dim, theta, b):
    def find_a_for_theta(a, p=p, dim=dim, theta=theta):
        temp_theta = []
        for i in range(100):
            rand_subspac0 = orth(np.random.randn(p, dim))
            rand_subspac1 = orth(np.random.randn(p, dim))
            rand_subspac2 = orth(np.random.randn(p, dim))
            temp_theta.append(subspace_angles(rand_subspac0 * (1 - a) + rand_subspac1 * a,
                                              rand_subspac0 * (1 - a) + rand_subspac2 * a).max())
        return (np.average(temp_theta) - theta)

    a = sc.optimize.bisect(find_a_for_theta, 0, 1)
    B = [b[0] * (1 - a) + b[i] * a for i in range(1, k + 1)]
    return (B)


# consider the following generative model for the data: zi ∼ U({1, .., K}), wi ∼ N(0, Id), xi|zi, wi ∼ N(Bziwi, σ2Ip)
def third_simulation(n, p, dim, B, k, theta):
    z = np.random.randint(0, k, n)
    w = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.diag(np.ones(dim)), size=n)
    X = np.zeros((n, p))
    for i in range(n):
        X[i,] = np.random.multivariate_normal(mean=np.array(np.dot(np.array(w[i, :]), B[z[i]].T)).flatten(),
                                              cov=np.diag(1* np.ones(p)))  # sigma value is missing/ change
    return (n, p, dim, theta, X, z, B)


# data simulation
def final_data_simulation(k):
    nn = [2 ** j for j in range(3, 11)]
    pp = [2 ** j for j in range(4, 8)]
    dd = [2 ** -j for j in range(1, 5)]
    tt = [10 ** -j for j in range(0, 3)]
    df = pd.DataFrame(columns=['n', 'p', 'dim', 'theta', 'X', 'z', 'B'])
    for p in pp:
        for d in dd:
            dim = int(d * p)
            b = first_simulation(p=p, dim=dim, k=k)
            for t in tt:
                theta = find_theta_max(p=p, dim=dim) * t
                if (t == 1):
                    a = 1
                    B = [b[0] * (1 - a) + b[i] * a for i in range(1, k + 1)]
                else:
                    B = second_simulation(p, k, dim, theta, b)
                for n in nn:
                    row = pd.Series(list(third_simulation(n=n, p=p, dim=dim, B=B, k=k, theta=theta)[0:7]),
                                    ["n", "p", "dim", "theta", "X", "z", "B"])
                    df = df.append([row], ignore_index=True)
    return (df)


# . After Algorithm (kmean and additional) clustering, which yields cluster identities ˆz1, .., zˆn,we estimate the sub-space of each cluster k by performing PCA on
# the points in this clusterand keeping the top d components as a basis for Bˆk for k = 1, ..,
def pca_subspace(df, i, dim):
    df_new = df[df['cluster'] == i].drop(['cluster'], axis=1)
    pca_components_number = len(df_new) - 1 if len(
        df_new) < dim else dim  # It is possible to get clusters of size smaller than d. you can for a generic cluster of m points, take the
    # unique sub-space of dimension m−1 passing through these points, and get a subspace with dimension less than d.
    pca = PCA(n_components=pca_components_number)
    pca.fit_transform(df_new)
    B_kmeans = pca.components_
    return (B_kmeans.T)


# apply cluster algo
def find_subspace(X, k, dim, algo):

    temp_df = pd.DataFrame(X)
    temp_df['cluster'] = algo(n_clusters=k).fit(X).labels_
    # ,algorithm='lasso_lars',gamma=50)  #learn about model parameters
    B = [pca_subspace(temp_df, i, dim) for i in range(k)]
    cluster= temp_df['cluster']
    return (B, cluster)


# The cost measures the angle between the original and estimated sub-spaces,with higher values achieved for smaller angle
def performance_measure1(k, B1, B2):
    all_per = list(it.permutations(range(k)))
    sum_cos_angles_all_per = np.zeros(len(all_per))
    for l, val in enumerate(all_per):
        for i in range(k):
            if B2[val[i]].shape[1] > 0:  # handling with empty clusters
                sum_cos_angles_all_per[l] += (math.cos(
                    subspace_angles(B1[i], B2[val[i]]).max())) ** 2  # use min or max????????????????
    cost_subspace = sum_cos_angles_all_per.max()
    return (cost_subspace)

    # WHAT ARE WE DOING WITH EMPTY CLUSTERS


# The cost in measures the fraction of points clustered correctly:

def performance_measure2(predicted_labels,true_labels):
    assert len(predicted_labels) == len(true_labels)
    data = {'predicted_labels': predicted_labels,'true_labels': true_labels}
    all_labels = pd.DataFrame(data, index=range(len(predicted_labels)))
    m = -1*np.array(all_labels.groupby(['predicted_labels','true_labels']).size().unstack(fill_value=0))
    indx, per = linear_sum_assignment(m)
    acc = -m[indx,per].sum()/len(all_labels)
    return (acc)


def all_process(k):
    df = final_data_simulation(k)
    #kmean_resluts = df.apply(lambda x: find_kmeans_subspace(x['X'], k, x['dim']), axis=1)
    kmean_resluts = df.apply(lambda x: find_subspace(x['X'], k, x['dim'],KMeans), axis=1)
    df['B_kmean'] = [pair[0] for pair in kmean_resluts]
    df['cluster_kmean'] = [pair[1] for pair in kmean_resluts]
    ensc_resluts = df.apply(lambda x: find_subspace(x['X'], k, x['dim'],ElasticNetSubspaceClustering), axis=1)
    df['B_ensc'] = [pair[0] for pair in ensc_resluts]
    df['cluster_ensc'] = [pair[1] for pair in ensc_resluts]
    return (df)

def binary_search(low, high, acc, p, dim, theta, k, iter, t):
    mid = (high + low) // 2
    value = find_accuracy_rate(mid, p, dim, theta, k, iter, t)
    if value <= acc and value >= 0:
        return mid
    elif value < 0:
        return binary_search(low, mid - 1, acc, p, dim, theta, k, iter, t)
    elif value > acc:
        return binary_search(mid + 1, high, acc, p, dim, theta, k, iter, t)
    else:
        return -1

def find_accuracy_rate(n, p, dim, theta, k, iter, t, sigma=0):
    accuracy_rate = []
    for r in range(iter):
        b = first_simulation(p, dim, k)
        if (t == 1):
            a = 1
            B = [b[0] * (1 - a) + b[i] * a for i in range(1, k + 1)]
        else:
            B = second_simulation(p, k, dim, theta, b)
        z = np.random.randint(0, k, n)
        w = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.diag(np.ones(dim)), size=n)
        X = np.zeros((n, p))
        for i in range(n):
            X[i,] = np.random.multivariate_normal(mean=np.array(np.dot(np.array(w[i, :]), B[z[i]].T)).flatten(),
                                                  cov=np.diag(sigma * np.ones(p)))  # sigma value is missing
        ensc_results = find_subspace(X, k, dim,ElasticNetSubspaceClustering)
        ensc_clusters = ensc_results[1]
        # kmeans_results = find_kmeans_subspace(X,k,dim)
        # kmeans_clusters = kmeans_results[1]
        # accuracy_rate.append(performance_measure2(k,z,kmeans_clusters))
        accuracy_rate.append(performance_measure2(k, z, ensc_clusters))
    avg_accuracy_rate = mean(accuracy_rate)
    return (avg_accuracy_rate - 0.5)


def find_b_constants(b_cons, df):
    optim_df = pd.DataFrame()
    for p in np.unique(df['p']):
        for t in np.unique(df['t']):
            n1, n2, n3, n4 = df['n_q'][(df['p'] == p) & (df['t'] == t)]
            sd = stdev([n1, n2, n3, n4])
            row = pd.Series([n1, n2, n3, n4, sd])
            optim_df = optim_df.append([row], ignore_index=False)
    optim_df['b_cons'] = b_cons
    new_df = optim_df.iloc[:, :4].apply(lambda x: (x - optim_df['b_cons']) / optim_df.iloc[:, 4], axis=0)
    # return (0 if (new_df.apply(lambda x: len(np.unique(round(x,2)))==1,axis=0)).all() else 1)
    # return new_df.apply(lambda x: len(np.unique(round(x,2)))==1,axis=0).sum()
    return new_df.apply(lambda x: variance(x), axis=0).sum()



def q_a():
    measure1_kmean = pd.DataFrame()
    measure2_kmean = pd.DataFrame()
    measure1_ensc = pd.DataFrame()
    measure2_ensc = pd.DataFrame()
    k = 4
    num_iters = 2
    for iter in range(num_iters):
        df = all_process(k)
        measure1_kmean.insert(iter, "", df.apply(lambda x: performance_measure1(k, x['B'], x['B_kmean']), axis=1), True)
        measure2_kmean.insert(iter, "", df.apply(lambda x: performance_measure2(x['z'], x['cluster_kmean']), axis=1),
                              True)
        measure1_ensc.insert(iter, "", df.apply(lambda x: performance_measure1(k, x['B'], x['B_ensc']), axis=1), True)
        measure2_ensc.insert(iter, "", df.apply(lambda x: performance_measure2(x['z'], x['cluster_ensc']), axis=1),
                             True)

    df['measure1_kmean'] = measure1_kmean.apply(lambda x: mean(x), axis=1)
    df['measure2_kmean'] = measure2_kmean.apply(lambda x: mean(x), axis=1)
    df['measure1_ensc'] = measure1_ensc.apply(lambda x: mean(x), axis=1)
    df['measure2_ensc'] = measure2_ensc.apply(lambda x: mean(x), axis=1)
    df['theta_degree'] = df.apply(lambda x: math.degrees(x['theta']), axis=1)
    df['t'] = list(np.repeat(np.array([1, 1 / 10, 1 / 100]), [8, 8, 8], axis=0)) * 16
    df['theta_degree'] = round(df['theta_degree'], 2)
    # df.to_csv('q1_df14.csv')
    # files.download('q1_df14.csv')
    df.head()

    # df2.to_csv('q1_df13.csv')
    # files.download('q1_df13.csv')
    # df2.head()

    # @title Default title text
    # if 'google.colab' in sys.modules:
    #     uploaded = files.upload()
    # df = pd.read_csv('q1_df12 (1).csv')
    df['cluster_kmean'] = df['cluster_kmean'].apply(lambda x: x.split('\n'))
    df['cluster_ensc'] = df['cluster_ensc'].apply(lambda x: x.split('\n'))
    df['B_kmean'] = df['B_kmean'].apply(lambda x: x.split('\n'))
    df['B_ensc'] = df['B_ensc'].apply(lambda x: x.split('\n'))

    all_measures = ["measure1_kmean", "measure2_kmean", "measure1_ensc", "measure2_ensc"]
    fig, axes = plt.subplots(8, 8, sharex=False, sharey=False, figsize=(32, 32))
    fig.suptitle('all measures for both clustering methods by p and dim', fontsize=24)
    pp = [2 ** j for j in range(4, 8)]
    dd = [2 ** -j for j in range(1, 5)]
    i = 0
    j = 0
    for p in pp:
        for d in dd:
            dim = int(d * p)
            for measure in all_measures:
                sns_df = df[(df['p'] == p) & (df['dim'] == dim)]
                sns_df = sns_df.pivot("theta_degree", "n", measure)
                sns.heatmap(sns_df, ax=axes[i, j])
                plt.subplots_adjust(wspace=1, hspace=1)
                # counter = counter+1
                axes[i, j].set_title('{a}: p= {b} ,dim= {c} '.format(a=measure, b=p, c=dim), fontsize=16)
                i = i if (j < 7) else i + 1
                j = j + 1 if (j < 7) else 0

    return df

def q_b(df):

    df2 = df.groupby(['p', 'dim', 'theta', 't']).size().reset_index()

    df2['d\p'] = df2['dim'] / df2['p']
    df2['n_q'] = np.repeat(0, len(df2))
    for row_no in range(len(df2)):
        df2['n_q'][row_no] = binary_search(4, 500, 0.1, df2['p'][row_no], df2['dim'][row_no], df2['theta'][row_no], 4,
                                           5,
                                           df2['t'][row_no])

    pp = np.unique(df2['p'])
    tt = np.unique(df2['t'])
    plt.figure(figsize=(13, 7))
    newcolors = ['#F00', '#F80', '#FF0', '#0B0', '#00F', '#50F', '#A0F', '#DC143C', '#00FFFF', '#00008B', '#008B8B',
                 '#B8860B']
    i = 0
    for p in np.unique(df2['p']):
        for t in np.unique(df2['t']):
            plt_df = df2[(df2['p'] == p) & (df2['t'] == t)]
            plt.plot(plt_df['d\p'], plt_df['n_q'], linewidth=4.0, c=newcolors[i], label="p= {a},t={b}".format(a=p, b=t))
            i = i + 1
            plt.xlabel("d/p", size=15)
            plt.ylabel("n0.5", size=15)
            plt.title("dim/p VS n0.5 in ENSC method", size=20)
            plt.legend(loc='upper left')
            positions = (1 / 16, 1 / 8, 1 / 4, 1 / 2)
            labels = ("0.0625", "0.125", "0.25", "0.5")
            plt.xticks(positions, labels)

    sc.optimize.bisect(find_b_constants, a=np.repeat(df2['n_q'].min(), 12), b=np.repeat(df2['n_q'].max(), 12),
                       args=(df2))

    f = minimize(find_b_constants, x0=np.random.randint(df2['n_q'].min(), df2['n_q'].max(), 12), args=(df2),
                 method="Nelder-Mead")
    xx = f['x']

    optim_df = pd.DataFrame()
    for p in np.unique(df2['p']):
        for t in np.unique(df2['t']):
            n1, n2, n3, n4 = df2['n_q'][(df2['p'] == p) & (df2['t'] == t)]
            sd = stdev([n1, n2, n3, n4])
            row = pd.Series([n1, n2, n3, n4, sd])
            optim_df = optim_df.append([row], ignore_index=False)
    optim_df['b_cons'] = xx
    new_df = optim_df.iloc[:, :4].apply(lambda x: (x - optim_df['b_cons']) / optim_df.iloc[:, 4], axis=0)
    new_df, new_df.apply(lambda x: variance(x), axis=0).sum()

    pp = np.unique(df2['p'])
    tt = np.unique(df2['t'])
    plt.figure(figsize=(13, 7))
    newcolors = ['#F00', '#F80', '#FF0', '#0B0', '#00F', '#50F', '#A0F', '#DC143C', '#00FFFF', '#00008B', '#008B8B',
                 '#B8860B']

    i = 0
    for p in np.unique(df2['p']):
        for t in np.unique(df2['t']):
            plt_df = df2[(df2['p'] == p) & (df2['t'] == t)]
            normalized_n_q = (plt_df['n_q'] - xx[i]) / stdev(plt_df['n_q'])
            plt.plot(plt_df['d\p'], normalized_n_q, linewidth=4.0, c=newcolors[i],
                     label="p= {a},t={b}".format(a=p, b=t))

            i = i + 1
            # plt.xlabel("d/p",size=15)
            # plt.ylabel("n0.5",size=15)
            # plt.title("dim/p VS n0.5 in ENSC method",size=20)
            # plt.legend(loc='upper left')
            # positions = (1/16,1/8,1/4,1/2)
            # labels = ("0.0625", "0.125", "0.25","0.5")
            # plt.xticks(positions, labels)

    pass


def main():
    df = q_a()
    #q_b(df)
    # apply algorithm


if __name__=='__main__':
    main()