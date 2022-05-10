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
    drop = keras.layers.Dropout(.3)(m_input)
    x = keras.layers.Dense(15,activation='tanh')(drop)
    for i in range(2):
        x = keras.layers.Dense(10,activation='ReLU')(x)
    drop = keras.layers.Dropout(.3)(x)
    m_output = keras.layers.Dense(2)(drop)

    opt = keras.optimizers.Adam(learning_rate=0.01)

    model = keras.Model(m_input, m_output, name="ffn")

    print(model.summary())
    return model, opt

def training_data(name):
    x = pd.read_hdf(f'data/forffn/{name}').to_numpy()
    return x

def raw_data(location, dim=2, error=False):
    if dim==2:
        x = pd.read_hdf(f'{location}.h5')
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
    elif dim==3:
        if error==True:
            x = pd.read_csv(f'{location}.csv')
            names=[]
            for i in x.columns:
                names.append(i)
            names = list(dict.fromkeys(names))
            for i in range(0, len(names)-13,6):
                y = x[f'{names[i]}'].to_numpy()   # x coordinate
                z = x[f'{names[i+1]}'].to_numpy() # y coordinate
                a = x[f'{names[i+2]}'].to_numpy() # z coordinate
                b = x[f'{names[i+3]}'].to_numpy() # errors
                y = np.reshape(y, (len(y),1))
                z = np.reshape(z, (len(z),1))
                a = np.reshape(a, (len(a),1))
                b = np.reshape(b, (len(b),1))
                arr = np.concatenate((y,z), axis=1)
                arr = np.concatenate((arr,a), axis=1)
                arr = np.concatenate((arr,b), axis=1)
                if i ==0:
                    out = arr
                else:
                    out = np.concatenate((out, arr), axis=1)
            return out
        else:
            x = pd.read_csv(f'{location}')
            names=[]
            for i in x.columns:
                names.append(i)
            names = list(dict.fromkeys(names))
            for i in range(0, len(names)-13,6):
                y = x[f'{names[i]}'].to_numpy()   # x coordinate
                z = x[f'{names[i+1]}'].to_numpy() # y coordinate
                a = x[f'{names[i+2]}'].to_numpy() # z coordinate
                y = np.reshape(y, (len(y),1))
                z = np.reshape(z, (len(z),1))
                a = np.reshape(a, (len(a),1))
                arr = np.concatenate((y,z), axis=1)
                arr = np.concatenate((arr,a), axis=1)
                if i ==0:
                    out = arr
                else:
                    out = np.concatenate((out, arr), axis=1)
            return out
    else:
        raise ValueError('Dimension must be 2 or 3')

def norm(x):
    for i in range(0,20,2):
        x[:,i] = x[:,i]/max(x[:,i])
        x[:,i+1] = x[:,i+1]/max(x[:,i+1])
    return x

def training(model,opt,x,y):
    model.compile(opt, loss="mae",metrics=['accuracy'])
    model.fit(x, y, epochs=9,batch_size=16,validation_split=0.02)

def repair(name):
    for i in range(0,20,2):
        model, opt = create_model()
        x = training_data(f'{name}.h5')
        x = norm(x)
        y = x[:,i:i+2]
        print(x.shape, y.shape)
        training(model,opt,x,y)
        if i ==0:
            x = raw_data(f'data/raw/{name}', dim=2)
            x = norm(x)
        else:
            x = pd.read_hdf(f"data/temp/top_{i-2}-{i-1}.h5").to_numpy()
            print(x.shape)
        pred = model.predict(x)
        minusone = x[:,i:i+2] <= 0
        print(f"Missing values in coulmns {i} and {i+1} before:",minusone.sum())
        for j in range(len(x)):
            if x[j][i] <= 0 and x[j][i+1] <= 0:
                x[j][i], x[j][i+1] = pred[j]
        minusone = x[:,i:i+2] <= 0
        print(f"Missing values in coulmns {i} and {i+1} after:",minusone.sum())
        print(x.shape)
        pd.DataFrame(x).to_hdf(f"data/temp/top_{i}-{i+1}.h5",index=False,key="d2c")

if __name__ == '__main__':
    init()
    name = '2d_1_bot'
    repair(name)