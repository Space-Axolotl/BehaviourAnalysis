import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

def init():
    print("\n"+"-"*30)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Tensorflow version:",tf.__version__)
    print("Python version: ", sys.version[:7])
    print("-"*30+"\n")

def create_model():
    m_input = keras.Input(shape=(20),batch_size=None,name='Input')
    drop = keras.layers.Dropout(.4)(m_input)
    x = keras.layers.Dense(15,activation='tanh')(drop)
    for i in range(2):
        x = keras.layers.Dense(10,activation='ReLU')(x)
    drop = keras.layers.Dropout(.3)(x)
    m_output = keras.layers.Dense(2)(drop)

    opt = keras.optimizers.Adam(learning_rate=0.01)

    model = keras.Model(m_input, m_output, name="ffn")

    print(model.summary())
    return model, opt

def read_data():
    x = pd.read_hdf('data/forffn/2d_ffn_2top.h5').to_numpy()
    return x

def prep_data(name):
    x = pd.read_hdf(f'data/raw/{name}')
    x.columns = x.columns.droplevel()
    names=[]
    # generate list of column names
    for i in x.columns:
        names.append(i[0])
    # delete duplicates
    names = list(dict.fromkeys(names))
    # loop through the data and create an output arr
    for i in range(0, len(names),2):
        y = x[f'{names[i]}'].drop(columns='likelihood').to_numpy()
        z = x[f'{names[i+1]}'].drop(columns='likelihood').to_numpy()
        arr = np.concatenate((y, z), axis=1)
        if i ==0:
            out = arr
        else:
            out = np.concatenate((out, arr), axis=1)
    print(names)
    print(out.shape)
    return out

def training(model,opt,x,y):
    model.compile(opt, loss="mae",metrics=['accuracy'])
    model.fit(x, y, epochs=9,batch_size=16,validation_split=0.02)

def norm(x):
    for i in range(0,20,2):
        x[:,i] = x[:,i]/825
        x[:,i+1] = x[:,i+1]/550
    return x

def repair():
    for i in range(0,20,2):
        model, opt = create_model()
        x = read_data()
        x = norm(x)
        y = x[:,i:i+2]
        print(x.shape, y.shape)
        training(model,opt,x,y)
        if i ==0:
            x = prep_data('2d_2_top.h5')
            x = norm(x)
        else:
            x = pd.read_hdf(f"data/temp/top2_{i-2}-{i-1}.h5").to_numpy()
        pred = model.predict(x)
        for j in range(len(x)):
            if x[j][i] <= 0 and x[j][i+1] <= 0:
                x[j][i], x[j][i+1] = pred[j]
        minusone = x[:,i:i+2] <= 0
        print(f"Missing values in coulmns {i} and {i+1} :",minusone.sum())
        pd.DataFrame(x).to_hdf(f"data/temp/top2_{i}-{i+1}.h5",index=False,key="d2c")

if __name__ == '__main__':
    init()
    repair()