from typing import no_type_check
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    # print(names)
    # print(out.shape)
    return out

x = prep_data('2d_2_top.h5')

def plotall():
    for i in range(0,20,2):
        plt.scatter(x[:,i],x[:,i+1],s=1)
    plt.show()

def animate():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim([500,855])
    ax.set_ylim([100,555])

    def animation(i):
        x = prep_data('2d_2_top.h5')
        ax.clear()
        ax.set_xlim([500,855])
        ax.set_ylim([100,555])
        for j in range(0,20,2):
            ax.scatter(x[i,j],x[i,j+1],s=15)

    animation = FuncAnimation(fig, func=animation, interval=12)
    plt.show()

# check how many points are in (-1,-1) 
# == 120419 this is 9,263% of datapoints 
# minusone = x == -1
# print(minusone.sum())
