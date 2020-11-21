import numpy  as np 
import pandas as pd
import matplotlib.pyplot as plt 
import torch
from sklearn.preprocessing import MinMaxScaler




def batch_for_few_shot(x, y , batch_size):
    # x shape : num , channel , height , weight
    # y shape : num 
    y_     = []
    target = []
    for i in range(len(y)):
        if (i+1) % batch_size == 0 :
            y_.append([0, 0])
            target.append(y[i])
        elif(y[i] == 0):
            y_.append([1, 0])
        elif(y[i] == 1):
            y_.append([0, 1])

    return torch.FloatTensor(x), torch.FloatTensor(np.array(y_)) ,torch.LongTensor(np.array(target))









def to_window(feature, label, time_step, image_type):
    #input : (days , feature)
    f = []
    
    for i in range( time_step , len(feature)):
        tmp = feature[i-time_step:i , :]
        if(image_type == 'GASF'):
            f.append(GASF(tmp))
        elif(image_type == 'GADF'):
            f.append(GADF(tmp))

    feature = np.array(f)
    label   = label[time_step:]

    return torch.FloatTensor(feature), torch.LongTensor(label)
    



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
    # input shape : (days, feature size)
    # print(data.shape)
    # output shape: (feature size, days, days)
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


def get_label(path):
    df = pd.read_csv(path)
    label = []
    for i in range(len(df)-3):
        if(np.mean(df["Close Price"].iloc[i:i+3]) - df["Close Price"].iloc[i] > 0.001):
            label.append(1)
        elif(df["Close Price"].iloc[i]  - np.mean(df["Close Price"].iloc[i:i+3]) > 0.001):
            label.append(0)
        else:
            label.append(-1)
    label.extend([-2,-2,-2])
    last = 0
    for i in range(len(label)):
        if(label[i] == -1):
            label[i] = last
        last = label[i]
    df['label'] = label
    df.to_csv("./data/S&P5002.csv")
    print(df)










def main():
    path = "./data/S&P500.csv"
    df = pd.read_csv(path, index_col = "Ntime")
    # print(df.shape)
    # print(df.info)
    value = GASF(df.iloc[:20,:].values)
    print(df.iloc[:30,:].shape)
    print(value.shape)
    input()
    plt.imshow(value[0])
    plt.show()



    value = GADF(df.iloc[:30,:].values)
    print(df.iloc[:30,:].shape)
    print(value.shape)
    input()
    plt.imsho
    plt.imshow(value[0])
    plt.show()





if __name__ == "__main__":
    main()