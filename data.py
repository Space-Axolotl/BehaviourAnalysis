import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def prep_data(location, dim=2, error=False):
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

def plotall(array, dim=2):
    if dim==2:
        for i in range(0,20,2):
            plt.scatter(array[:,i],array[:,i+1],s=1)
        plt.show()
    elif dim==3:
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(0,30,3):
            ax.scatter3D(array[:1000,i],array[:1000,i+1],array[:1000,i+2],s=1)
        plt.show()
    else:
        raise ValueError('Dimension must be 2 or 3')

def remove_nan(array,name,create=False):
    new = np.array([array[-1]])
    for i in range(0,len(array)):
        comparison = False
        for j in range(0,len(array[i])):
            if array[i][j] == -1.0:
                comparison=True
                break
        if comparison==False:
            new = np.append(new, [array[i]],axis=0)
    if create == True:
        pd.DataFrame(new).to_hdf(f"~/BehaviourAnalysis/data/forffn/{name}.h5",index=False,key="d2")
    print(new.shape)



if __name__=='__main__':
    name = '2d_2_bot'

    x = prep_data(f'data/raw/{name}', dim=2)
    # x = prep_data('data/raw/3d_1_top.csv', dim=3, error=False)

    # plotall(x, dim=2)
    remove_nan(x,name,create=True)



