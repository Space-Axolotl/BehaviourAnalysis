import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def data(name):
    x = pd.read_hdf(f'data/clean/{name}').to_numpy()
    return x

def create_model(dshape):
    enc_in = keras.Input(shape=(dshape),name='Input')
    x = keras.layers.Dense(15,activation='sigmoid')(enc_in)
    x = keras.layers.Dropout(.1)(x)
    x = keras.layers.Dense(10,activation='relu')(x)
    x = keras.layers.Dense(5)(x)
    enc_out = keras.layers.Dense(2)(x)
    dec_in = keras.layers.Dense(5)(enc_out)
    x = keras.layers.Dense(10,activation='relu')(dec_in)
    x = keras.layers.Dropout(.1)(x)
    x = keras.layers.Dense(15,activation='sigmoid')(x)
    dec_out = keras.layers.Dense(dshape)(x)

    opt = keras.optimizers.Adam(learning_rate=.00087)

    enc = keras.Model(enc_in, enc_out, name='encoder')
    autoenc = keras.Model(enc_in, dec_out, name='autoencoder')

    print(autoenc.summary())
    return autoenc, enc, opt

def norm(x):
    for i in range(0,20,2):
        x[:,i] = x[:,i]/max(x[:,i])
        x[:,i+1] = x[:,i+1]/max(x[:,i+1])
    return x
x = data('2d_1_top_c.h5')
autoenc, enc, opt = create_model(x.shape[1])

x = norm(x)

autoenc.compile(opt, loss="mae",metrics=['accuracy'])
autoenc.fit(x, x, epochs=20,validation_split=0.2)

enc.compile(opt, loss="mae",metrics=['accuracy'])
pred = enc.predict(x)

autoenc.save('models/autoencenc_top1_20.h5')
enc.save('models/enc_top1_20.h5')

plt.scatter(pred[:,0],pred[:,1],s=1)
plt.show()