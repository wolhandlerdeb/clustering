#2. Real Data Analysis. Read, make diagnostic plots and cluster a real dataset.
import time
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from utils.mnist_reader import load_mnist
from cluster.selfrepresentation import ElasticNetSubspaceClustering,SparseSubspaceClusteringOMP
from scipy.optimize import linear_sum_assignment

def calculate_angle(p1, p2):
    p1_u = p1 / np.linalg.norm(p1)
    p2_u = p2 / np.linalg.norm(p2)
    return (np.arccos(np.clip(np.dot(p1_u, p2_u), -1.0, 1.0)))

def preprocess_substract_mean(X, y):
    labels = np.unique(y)
    X_processed= X.copy()
    for l in labels:
        mean = np.average(X_processed[y == l], 0)
        X_processed[y == l] = X_processed[y == l]- mean

    return X_processed


def performance_measure(predicted_labels,true_labels):
    assert len(predicted_labels) == len(true_labels)
    data = {'predicted_labels': predicted_labels,'true_labels': true_labels}
    all_labels = pd.DataFrame(data, index=range(len(predicted_labels)))
    m = -1*np.array(all_labels.groupby(['predicted_labels','true_labels']).size().unstack(fill_value=0))
    indx, per = linear_sum_assignment(m)
    acc = -m[indx,per].sum()/len(all_labels)
    return (acc)


def q_a(X,y):
    # (10pt) Download the fashion MNIST [5] dataset from here.
    # The training set contains 60, 000 grayscale images of size 28×28 pixels representing 10 different types of clothes.
    # Our goal is to cluster the images into the clothes types without supervision.
    # To make the problem more difficult for standard clustering algorithms, subtract the mean from every class,
    # such that all 10 cluster centers are at the origin.
    # Run PCA on the dataset and plot the projection on the first 2 principal components,
    # with each class marked in a different color/symbol. Are the classes well-separated?

    X_processed = preprocess_substract_mean(X, y)
    pca = PCA(2)  # project from 64 to 2 dimensions
    projected = pca.fit_transform(X_processed)
    print(X_processed)
    print(projected.shape)
    plt.scatter(projected[:, 0], projected[:, 1],
                c=y, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('tab10', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    # plt.colorbar();
    plt.show()


def q_b_v2(X,y):
    # Sample at least 5000 pairs of points from the same class and 5000 pairs of points from different classes,
    labels = np.unique(y)
    n=5000
    cos_theta_in_all = np.empty( shape=(0, 0) )
    cos_theta_out_all = np.empty( shape=(0, 0) )
    num_labels = len(labels)
    rand_indx1 = random.choices(range(len(X)), k=int(n))
    #rand_indx2 = list(pd.Series(rand_indx1).apply(lambda x: random.choices(y.index[y ==y[x]])))
    rand_indx2 = [random.choice(np.where(y==y[idx])[0]) for idx in rand_indx1]

    #rand_indx2 = [j[0] for j in rand_indx2]
    #rand_indx3 = list(pd.Series(rand_indx1).apply(lambda x: random.choices(y.index[y !=y[x]])))
    rand_indx3 = [random.choice(np.where(y!=y[idx])[0]) for idx in rand_indx1]

    #rand_indx3 = [j[0] for j in rand_indx3]
    points_in_1 = X[rand_indx1,:]
    points_in_2 = X[rand_indx2,:]
    points_out_1 = X[rand_indx3,:]
    #compute the angle between every pair of points
    theta_in_all =  [calculate_angle(points_in_1[i,:],points_in_2[i,:]) for i in range(len(points_in_1))]
    theta_out_all = [calculate_angle(points_in_1[i,:],points_out_1[i,:]) for i in range(len(points_in_1))]
    # Plot the distribution of between-cluster angles and within cluster angles.
    sns.distplot(theta_in_all,hist=True)
    sns.distplot(theta_out_all,hist=True)
    plt.legend(labels=['theta in', 'theta out'])
    plt.show()


def q_b(X,y):
    #(b) (10pt) Sample at least 5000 pairs of points from the same class
    # and 5000 pairs of points from different classes,
    # and compute the angle between every pair of points.
    # Plot the distribution of between-cluster angles and within cluster angles.
    # Do you see a difference between the distributions?
    labels = np.unique(y)
    #n = 50
    n=5000
    cos_theta_in_all = np.empty( shape=(0, 0) )
    cos_theta_out_all = np.empty( shape=(0, 0) )
    num_labels = len(labels)
    for l in labels:
        points_in_1 = X[y == l][0:int(n/(num_labels)),:]
        points_in_2 = X[y == l][int(n/(num_labels)):int(2*n/num_labels),:]
        rand_idx = random.choices(range(len(X[y != l])), k=int(n/num_labels))
        points_out = X[y != l][rand_idx,:]

        # angle calculation
        cos_theta_in_tmp =np.transpose([(points_in_1*points_in_2).sum(axis=1)])
        cos_theta_in = np.divide(cos_theta_in_tmp, (
                    np.linalg.norm(points_in_1, axis=1, keepdims=True) * np.linalg.norm(points_in_2, axis=1,
                                                                                        keepdims=True)))

        cos_theta_in_all =np.append(cos_theta_in_all,cos_theta_in)
        cos_theta_out_tmp =np.transpose([(points_in_1*points_out).sum(axis=1)])
        cos_theta_out = np.divide(cos_theta_out_tmp, (
                np.linalg.norm(points_in_1, axis=1, keepdims=True) * np.linalg.norm(points_out, axis=1,
                                                                                    keepdims=True)))
        cos_theta_out_all =np.append(cos_theta_out_all, cos_theta_out)


    theta_in_all = np.arccos(cos_theta_in_all)
    theta_out_all= np.arccos(cos_theta_out_all)
    sns.distplot(theta_in_all,hist=True)
    sns.distplot(theta_out_all,hist=True)
    plt.legend(labels=['theta in', 'theta out'])
    plt.show()


def q_c(X,y):
    # (10pt) Perform PCA for each class separately, and plot for each class the proportion of variance explained vs.
    # the number of components, ordered from the first PC until the last. Repeat but now with PCA for the entire dataset.
    # What number of components would you take for further analysis?
    labels = np.unique(y)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for l in labels:
        pca = PCA()  # project from 64 to 2 dimensions
        projected = pca.fit_transform(X[y==l])
        exp_var_ratio = pca.explained_variance_ratio_
        ax1.plot(exp_var_ratio,label=f'class {l}')
        ax2.plot(np.cumsum(pca.explained_variance_ratio_),label=f'class {l}')
    ax1.set_title("Explained Variance per class")
    ax2.set_title("Cumulated Explained Variance per class")
    ax1.legend()
    ax2.legend()
    fig1.show()
    fig2.show()

    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    pca = PCA()  # project from 64 to 2 dimensions
    projected = pca.fit_transform(X)
    exp_var_ratio = pca.explained_variance_ratio_
    ax3.plot(exp_var_ratio)
    ax4.plot(np.cumsum(exp_var_ratio))
    ax3.set_title("Explained Variance Global")
    ax4.set_title("Cumulated Explained Variance Global")
    fig3.show()
    fig4.show()

    pca = PCA(0.9)
    pca.fit_transform(X)
    print(f"The number of components necessary to explain 90% of the data is : {pca.n_components_}")




def q_d(X,y):
    #(d) (10pt) Run the following algorithms on your dataset:
    #i. K-means with K = 10.
    #ii. PCA with the number of components chosen based on (c.), followed by K-means with K = 10
    # on the projection to the top components.
    #iii. A subspace clustering algorithm of your choice, where you can set the number of clusters to the correct one, 10.
    #For each algorithm, compute and report the clustering accuracy from eq. (6). Explain your results.
    labels = np.unique(y)

    K=10
    #i
    kmeans = KMeans(n_clusters=K).fit(X)

    kmeans_acc =performance_measure(kmeans.labels_,y)

    #ii
    num_components = 85
    #num_components = PCA(0.9).n_components
    pca = PCA(n_components=num_components)
    pca_X =pca.fit_transform(X)
    kmeans_after_PCA = KMeans(n_clusters=K).fit(pca_X)

    kmeans_pca_acc = performance_measure(kmeans_after_PCA.labels_,y)

    # iii

    model_ssc_omp = SparseSubspaceClusteringOMP(n_clusters=K,
                                                affinity='symmetrize',
                                                n_nonzero=5,thr=1.0e-5)
    model_ssc_omp.fit(X)
    ssc_omp_acc = performance_measure(model_ssc_omp.labels_, y)


    #model_ensc = ElasticNetSubspaceClustering(n_clusters=K, algorithm='spams', gamma=500)
    #model_ensc.fit(X)
    #ensc_acc =performance_measure(model_ensc.labels_, y)

    print(f'kmeans acc is: {kmeans_acc} ,'
          f' pca followed by kmeans acc is : {kmeans_pca_acc},'
         # f' ensc acc is {ensc_acc}'
          f' ssc omp acc is {ssc_omp_acc}')


def main():
    X_train, y_train = load_mnist('data/fashion', kind='train')
    X_test, y_test = load_mnist('data/fashion', kind='t10k')

    X_train =X_train.astype(np.uint)
    y_train =y_train.astype(np.uint)
    X_test = X_test.astype(np.uint)
    y_test = y_test.astype(np.uint)
    #q_a(X_train, y_train)
    #q_b(X_train, y_train)
    #q_b_v2(X_train, y_train)
    #q_c(X_train, y_train)
    q_d(X_train,y_train)

if __name__ == '__main__':
    main()





