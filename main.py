def init():
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
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Tensorflow v",tf.__version__)
    print("Python version: ", sys.version)

if __name__ == "__main__":
    init()

def data():
    pass

def dreduct():
    pass

def clustering():
    pass