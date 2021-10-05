import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

def create_model():
    m_input = keras.Input(shape=(1,20),name='Input')
    x = keras.layers.LSTM(10,input_shape=(1,20),return_sequences=True)(m_input)
    x = keras.layers.LSTM(10,return_sequences=True)(x)
    x = keras.layers.LSTM(10,return_sequences=False)(x)
    x = keras.layers.Dense(5)(x)
    m_output = keras.layers.Dense(2)(x)

    opt = keras.optimizers.Adam(learning_rate=0.01)

    model = keras.Model(m_input, m_output, name="ffn")

    print(model.summary())
    return model, opt

def norm(x):
    for i in range(0,20,2):
        x[:,i] = x[:,i]/825
        x[:,i+1] = x[:,i+1]/550
    return x

def trainning_set():
    x = pd.read_hdf('data/forffn/2d_ffn_2top.h5').to_numpy()
    x = norm(x)
    y = x[:,:2]
    x = np.reshape(x,(-1,1,20))
    y = np.reshape(y,(-1,1,2))
    return x,y

def test_set():
    x = pd.read_hdf('data/raw/2d_2_top.h5').to_numpy()
    x = norm(x)
    x = np.reshape(x,(-1,1,20))
    return x

def train(model, opt, x, y):
    model.compile(opt, loss="mae",metrics=['accuracy'])
    model.fit(x, y, epochs=10,batch_size=16,validation_split=0.02)



for i in range(0,20,2):
    x, y = trainning_set()
    model, opt = create_model()
    train(model,opt,x,y)
    if i ==0:
        x = test_set()
    else:
        x = pd.read_hdf(f"data/temp/top2_{i-2}-{i-1}.h5").to_numpy()
        x = np.reshape(x,(-1,1,20))
    pred = model.predict(x)
    print(pred.shape, "hey its me mario",x.shape)
    pred = np.reshape(pred,(-1,2))
    x = np.reshape(x,(-1,2))
    for j in range(0,len(pred)):
        if x[j][i] <= 0 and x[j][i+1] <= 0:
            x[j][i], x[j][i] = pred[j][0],pred[j][1]
    minusone = x[:,i:i+2] <= 0
    print(f"Missing values in coulmns {i} and {i+1} :",minusone.sum())
    pd.DataFrame(x).to_hdf(f"data/temp/top2_{i}-{i+1}.h5",index=False,key="d2c")