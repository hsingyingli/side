import numpy  as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler




def Ploar_Coordinate(data, bound = 0):
    '''
    input shape  : (days , feature size)
    step1: normalize
    step2: x' = arcos(x)
    output shape : (days, feature size)
    '''
    
    # step1
    scaler         = MinMaxScaler((bound,1)).fit(data)
    normalize_data = scaler.transform(data)
    
    for i in range(normalize_data.shape[0]):
        for j in range(normalize_data.shape[1]):
            if(normalize_data[i][j] > 1.0):
                normalize_data[i][j] = 1.0
            if(normalize_data[i][j] < bound):
                normalize_data[i][j] =bound


    # step2
    ploar_data = np.arccos(normalize_data)
   
    return ploar_data


def GASF(data):
    # input shape: (days, feature size)
    # print(data.shape)
    gasf = []
    ploar_data = Ploar_Coordinate(data, 0)

    for i in range(ploar_data.shape[1]):
        data = ploar_data[:,i].reshape(-1)
        tmp = []
        for row in range(data.shape[0]):
            for col in range(data.shape[0]):
                tmp.append(np.cos(data[row]) * np.cos(data[col]) - np.sin(data[row]) * np.sin(data[col]))
        tmp = np.array(tmp).reshape(data.shape[0],data.shape[0])
        gasf.append(tmp)
    
    gasf = np.array(gasf)
   
    return gasf
def GADF(data):
    # input shape: (days, feature size)
    # print(data.shape)
    gadf = []
    ploar_data = Ploar_Coordinate(data, 0)

    for i in range(ploar_data.shape[1]):
        data = ploar_data[:,i].reshape(-1)
        tmp = []
        for row in range(data.shape[0]):
            for col in range(data.shape[0]):
                tmp.append(np.sin(data[row]) * np.cos(data[col]) - np.cos(data[row]) * np.sin(data[col]))
        tmp = np.array(tmp).reshape(data.shape[0],data.shape[0])
        gadf.append(tmp)
    
    gadf = np.array(gadf)
    
    return gadf






def main():
    path = "./data/S&P500.csv"
    df = pd.read_csv(path, index_col = "Ntime")
    # print(df.shape)
    # print(df.info)
    value = GASF(df.iloc[:30,:].values)
    plt.imshow(value[0])
    plt.show()



    value = GADF(df.iloc[:30,:].values)
    plt.imshow(value[0])
    plt.show()


























if __name__ == "__main__":
    main()