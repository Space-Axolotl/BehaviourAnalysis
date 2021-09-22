import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

def init():
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Tensorflow version:",tf.__version__)
    print("Python version: ", sys.version[:7])


