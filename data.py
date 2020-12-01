import numpy  as np 
import pandas as pd
import matplotlib.pyplot as plt 
import torch
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler




def batch_for_few_shot(data, rise, fall, batch, start, end):
    # data: dict:  feature, label, date
    # rise: dict:  feature, label, date
    # fall: dict:  feature, label, date
    # train start: 2009-06
    # train end  : 2015-05
    # batch : 41    20 for rise, 20 for fall, 1 for target
    x = []
    y = []
    t = []
    
    
    start_index = np.where(data["date"] >= start)[0][0]
    end_index   = np.where(data["date"] <= end)[-1][-1]


    for i in range(end_index - start_index +1):
        tmp_end_date = data["date"][start_index]
        tmp_start_date = pre_year(tmp_end_date)
        
        # rise_index = np.where(rise["date"]>= tmp_start_date and rise["date"] < tmp_end_date)[0]
        # fall_index = np.where(fall["date"]>= tmp_start_date and fall["date"] < tmp_end_date)[0]



        rise_index = [ int(i) for i in np.where(rise["date"]>= tmp_start_date)[0] if i in np.where(rise["date"] < tmp_end_date)[0]]
        fall_index = [ int(i) for i in np.where(fall["date"]>= tmp_start_date)[0] if i in np.where(fall["date"] < tmp_end_date)[0]]
        

        rise_index = np.random.choice(rise_index,int((batch-1)/2), replace=False)
        fall_index = np.random.choice(fall_index,int((batch-1)/2), replace=False)

        

        rise_feature = rise["feature"][rise_index]
        rise_label   = rise["label"][rise_index]    # [0, 1]

        fall_feature = fall["feature"][fall_index]
        fall_label   = fall["label"][fall_index]    #[1, 0]

        target_feature = data["feature"][start_index]
        target_label   = data["label"][start_index] #[1 or 0]

        for i in range(int((batch-1)/2)):
            x.append(rise_feature[i])
            x.append(fall_feature[i])
            y.append(rise_label[i])
            y.append(fall_label[i])

        x.append(target_feature)
        y.append(np.array([0,0]))

        t.append(target_label)


        start_index += 1 


    return torch.FloatTensor(x), torch.FloatTensor(np.array(y)) ,torch.LongTensor(np.array(t))

def pre_year(date):
    year  = int(date[:4])
    month = int(date[5:7])
    month -= 6

    if(month<=0):
        month += 12
        year  -= 1


    d = str(year) + "-"
    if(month <10):
        d += "0" + str(month)
    else:
        d += str(month)

    d += date[7:]

    
    return d


def get_rise_fall(data):
    x = data["feature"]
    y = data["label"]
    d = data["date"]
    
    rise = {}
    fall = {}
    rise["feature"], rise["label"], rise["date"] = [], [], []
    fall["feature"], fall["label"], fall["date"] = [], [], []

    for i in range(len(y)):
        if(y[i] == 1):
            rise["feature"].append(x[i])
            rise["label"].append(np.array([0,1]))
            rise["date"].append(d[i])
        elif(y[i] == 0):
            fall["feature"].append(x[i])
            fall["label"].append(np.array([1,0]))
            fall["date"].append(d[i])

    rise["feature"] = np.array(rise["feature"])
    rise["label"] = np.array(rise["label"])
    rise["date"] = np.array(rise["date"])

    fall["feature"] = np.array(fall["feature"])
    fall["label"] = np.array(fall["label"])
    fall["date"] = np.array(fall["date"])
    
    return rise, fall






def to_window(feature, label, date, time_step, image_type):
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
    
    date    = [(str(i)[:10]) for i in date[time_step:] ]
    
    return feature, label, np.array(date)
    



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
    pre_year("2014-01-01")