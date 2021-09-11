# General imports
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from numba import jit
from pandas.core.arrays.sparse import dtype
# Dimensionality reduction
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

# Clustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import KMeans

def init():
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Tensorflow version:",tf.__version__)
    print("Python version: ", sys.version[:7])

def data(name, num):
        df = pd.read_hdf(f'{name}_{num}.h5').to_numpy()[:200000]
        return df

def dreduct(method,name,num):
    if method=='PCA':
        pca = PCA(n_components=2)
        X = data(name,num)
        pcs = pca.fit_transform(X)
        print("Principal Component accuracy: ",sum(pca.explained_variance_ratio_))
        # print(pca.singular_values_)
        return pcs
    elif method=='Isomap':
        embedding = Isomap(n_components=2)
        X = data(name,num)
        X_transformed = embedding.fit_transform(X)
        print("Manifold embeding succesful :)")
        return X_transformed
    else:
        raise ValueError('A very specific bad thing happened. (incorrectly specified DATA REDUCTION method)')
        
def clustering(method,X):
    if method == 'K-means':
        kmeans = KMeans(n_clusters=6, random_state=42).fit(X)
        print('clustering complete')
        return kmeans.labels_
    elif method=='DBSCAN':
        clustering = DBSCAN(eps=3, min_samples=2).fit(X)
        print('DBSCAN clustering complete')
        return clustering.labels_
    elif method=='BIRCH':
        brc = Birch(n_clusters=None)
        brc.fit(X)
        labels = brc.predict(X)
        print('BIRCH clustering complete')
        return labels
    else:
        raise ValueError('A very specific bad thing happened. (incorrectly specified CLUSTERING method)')

def graph(X,labels,title):
    fig, ax = plt.subplots()
    for label in np.unique(labels):
        i = np.where(labels == label)
        ax.scatter(X[i,0], X[i,1], label=label,marker='.',s=1)
    # ax.legend()
    # plt.scatter(X[:,0],X[:,1],c=labels,s=1)
    plt.title(title[0])
    plt.xlabel(title[1])
    plt.ylabel(title[2])
    # plt.show()
    plt.savefig(f'{title[0]}{title[1]}.png')


if __name__ == "__main__":
    init()

# X = dreduct('Isomap','data/c2d_data','1')
# labels = clustering('K-means',X)
# graph(X,labels,['Title','x_label','y_label'])

def OmegaDataset(foldername,filename):
    X = np.array([])
    for i in range(55):
        Y = pd.read_hdf(f'{foldername}/{filename}{i}.h5').to_numpy()
        X = np.append(X,Y)
    X = np.reshape(X,(-1,20))
    # print(X.shape,X[:76682]==data('data/c2d_data','0'))
    pd.DataFrame(X).to_hdf("OmegaData.h5",index=False,key="d2c")

# OmegaDataset('~/Rats/data','c2d_data_')

def experiments():
    exlist = [['PCA','K-means'],['PCA','DBSCAN']]
    for i in exlist:
        print(i[0])
        X = dreduct(f'{i[0]}','OmegaData','0')
        labels = clustering(f'{i[1]}',X)
        graph(X,labels,[f'{i[0]}',f'{i[1]}','centered data'])
experiments()