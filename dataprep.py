from typing import no_type_check
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numba
from numba import njit 

def init():
    return 0

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

def speed():
    new=np.array([x[0]])
    for i in range(x.shape[0]):
        r=True
        for j in range(0,20,2):
            if x[i][j] < 0 and x[i][j+1] < 0:
                r=False
        if r == True:
            new = np.append(new,[x[i]],0)

    print(new.shape)
    pd.DataFrame(new).to_hdf("~/BehaviourAnalysis/data/forffn/2d_ffn_1bot.h5",index=False,key="d2")


if __name__== '__main__':
    init()
    x = prep_data('2d_1_bot.h5')
    # speed()
    # plotall()
    # animate()


# minusone = x == -1
# print(minusone.sum())
# check how many points are in (-1,-1) 
# == 120419 this is 9,263% of datapoints 
