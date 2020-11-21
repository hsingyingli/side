import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
import random

def MinMax(inputs,lower_bound,upper_bound):
    scaler= MinMaxScaler(feature_range = (lower_bound,upper_bound)).fit(inputs)
    output = scaler.transform(inputs)
    
    return scaler,output

def inverse_MinMax(inputs,scaler):
    output = scaler.inverse_transform(inputs)
    return output


def train_test_split(x,y,time_step,num_channels, N, K):
    feature = []
    label   = []
    f       = []


    

    for i in range(num_channels):
        temp = x[:,i]
        f.append(temp)
    f = np.array(f)

    for i in range(time_step , len(x)+1):
        temp = f[:,i-time_step:i]    
        feature.append(temp)

    for i in range(time_step,len(y)+1):
        temp = y[i-1:i]   
        label.append(temp)
     

    feature = np.array(feature)
    if(feature.shape[2] == 1):
        feature = feature.reshape(-1,num_channels)
    
    label   = np.array(label).reshape(-1)
    

    train_feature = []
    train_label   = []
    test_feature  = []
    test_label    = []


    bear_index = []
    bull_index = []


    # for i in range(len(feature)):
    #     if len(bear_index) == K and len(bull_index) == K:
    #         for j in range(K):
    #             train_feature.append(feature[bear_index[j]])
    #             train_feature.append(feature[bull_index[j]])
    #             train_label.append(0)
    #             train_label.append(1)

    #         train_feature.append(feature[i])
    #         train_label.append(label[i])
    #         bear_index = []
    #         bull_index = []
    #         continue

    #     if label[i] == 1:
    #         bull_index.append(i)
            
    #         while len(bull_index) > K:
    #             bull_index.pop(0)

    #     else:
    #         bear_index.append(i)

    #         while len(bear_index) > K:
    #             bear_index.pop(0)

    for i in range(len(feature)):
        if len(bear_index) == K and len(bull_index) == K:
            for j in range(K):
                train_feature.append(feature[bear_index[j]])
                train_feature.append(feature[bull_index[j]])
                train_label.append(0)
                train_label.append(1)

            train_feature.append(feature[i])
            train_label.append(label[i])

        if label[i] == 1:
            bull_index.append(i)
            
            while len(bull_index) > K:
                bull_index.pop(0)

        else:
            bear_index.append(i)

            while len(bear_index) > K:
                bear_index.pop(0)


    length = int(len(train_feature)*0.9/(N*K+1))*(N*K+1)
    

    test_feature = train_feature[length:]
    test_label   = train_label[length:]

    train_feature = train_feature[:length]
    train_label = train_label[:length]
    # print(len(train_feature))
    # print(len(test_feature))
    # input()


    # split bear and bull
    # bull_feature = [] 
    # bear_feature = []
    

    # for (_x, _y) in zip(feature, label):
        
    #     if _y == 1:
    #         bull_feature.append(_x)
    #     else:
    #         bear_feature.append(_x)

    # ran = random.sample(range(0, len(bear_feature)-1), len(bull_feature)-len(bear_feature))

    # for i in ran:
    #     bear_feature.append(bear_feature[i])   

    # train_feature = []
    # train_label = []
    # test_feature = []




    # for i in range(int(len(bull_feature)*0.9)):

    #     train_feature.append(bull_feature[i])
    #     train_feature.append(bear_feature[i])

    #     train_label.append(1)
    #     train_label.append(0)

    
    # for i in range(0,len(train_feature)-1,3):
    #     ran = np.random.randint(0,10,1)

    #     if(ran%2 == 0):
    #         temp1             ,  temp2           = train_feature[i], train_label[i]
    #         train_feature[i]  , train_label[i]   = train_feature[i+1], train_label[i+1]
    #         train_feature[i+1], train_label[i+1] = temp1, temp2

    #         try:
    #             temp1, temp2 = train_feature[i+2], train_label[i+2]
    #             train_feature[i+2]  , train_label[i+2]   = train_feature[i+5], train_label[i+5]
    #             train_feature[i+5], train_label[i+5] = temp1, temp2
    #         except:
    #             pass

    

    # test_feature = bull_feature[int(len(bull_feature)*0.9):] + \
    #                 bear_feature[int(len(bear_feature)*0.9):]   
    # test_label = [1 for i in range(len(bull_feature)-int(len(bull_feature)*0.9))]
    # test_label += [0 for i in range(len(bear_feature)-int(len(bear_feature)*0.9))]
    


    train_feature = np.array(train_feature)
    train_label   = np.array(train_label)
    test_feature  = np.array(test_feature)
    test_label    = np.array(test_label)
    
    
    return  train_feature , train_label , test_feature , test_label



def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def R(y_true, y_pred):
    y_true = (y_true - np.mean(y_true)) / np.std(y_true)
    y_pred = (y_pred - np.mean(y_pred)) / np.std(y_pred)
    return  np.mean(np.multiply(y_true,y_pred))

def TheilU(y_true, y_pred):
    x = np.sqrt(np.mean(np.multiply((y_true - y_pred),(y_true - y_pred))))
    y = np.sqrt(np.mean(np.multiply(y_true,y_true))) + np.sqrt(np.mean(np.multiply(y_pred,y_pred)))
    return x/y

def inverse(data, mean_, var_):
    for i in range(len(data)):
        data[i] = var_[i] * data[i] + mean_[i]
    return data


def add_one_hot(features, labels, last_index):
    zero = np.array([0, 0])
    temp = np.array([])
    fall = np.array([1,0])
    rise = np.array([0,1])
    for i in range(len(features)):
        if (i in last_index):
            temp = np.concatenate((temp, zero), axis = 0)
        else:
            if(labels[i] == 0):
                temp = np.concatenate((temp,fall), axis = 0)
            elif(labels[i] == 1):
                temp = np.concatenate((temp,rise), axis = 0)
            
    temp = temp.reshape(-1, 2)
    features = np.concatenate((features.detach().numpy() , temp), axis = 1)
    
    return torch.Tensor(features)

def batch_for_few_shot(y, batch):
    y = y.detach().numpy()
    _y = np.array([])
    last_target = np.array([])

    for i in range(y.shape[0]):
        if( (i + 1) % batch == 0):
            last_target = np.concatenate((last_target, [y[i]]), axis = 0)
   
    return torch.LongTensor(last_target.reshape(-1))
            

