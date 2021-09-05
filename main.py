# General imports
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        df = pd.read_hdf(f'data/{name}_{num}.h5')
        return df

def dreduct(method,name,num):
    if method=='PCA':
        pca = PCA(n_components=2)
        X = data(name,num)
        pca.fit(X)
        print("Principal Component accuracy: ",sum(pca.explained_variance_ratio_))
        # print(pca.singular_values_)
        return pca.singular_values_
    elif method=='Isomap':
        embedding = Isomap(n_components=2)
        X = data(name,num)
        X_transformed = embedding.fit_transform(X)
        print("Manifold embeding succesful :)")
        return X_transformed
    else:
        raise ValueError('A very specific bad thing happened. (incorrectly specified DATA REDUCTION method)')

def clustering(method,name,num):
    if method == 'K-means':
        X = data(name,num)
        kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
        print('clustering complete')
        return kmeans.labels_
    elif method=='DBSCAN':
        X = data(name,num)
        clustering = DBSCAN(eps=3, min_samples=2).fit(X)
        print('DBSCAN clustering complete')
        return clustering.labels_
    elif method=='BIRCH':
        brc = Birch(n_clusters=None)
        X = data(name,num)
        labels = brc.fit_predict(X)
        print('BIRCH clustering complete')
        return labels
    else:
        raise ValueError('A very specific bad thing happened. (incorrectly specified CLUSTERING method)')

def graph(X):
    
    pass

if __name__ == "__main__":
    init()
dreduct('PCA','c2d_data','1')
