import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from process import add_one_hot
from collections import OrderedDict
import math


class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels,dilation, kernal_size = 2):
        super(CausalConv, self).__init__()
        self.dilation = dilation
        self.padding = (kernal_size-1)*dilation
        self.conv    = nn.Conv1d(in_channels, out_channels, kernal_size, 
                                padding = self.padding, stride = 1, dilation = dilation)
    def forward(self, x):
        # input shape (batch_size , channel , time_step)
        out = self.conv(x)
        return out[:, :, :-1*self.dilation]




class DenseBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dilation):
        super(DenseBlock, self).__init__()
        self.h1      = CausalConv(in_channel, out_channel, dilation)
        self.h2      = CausalConv(in_channel, out_channel, dilation)
        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

    def forward(self, x):
        # input shape (batch_size , channel , time_step)
        x1  = self.sigmoid(self.h1(x))
        x2  = self.tanh(self.h2(x))
        out = x1 * x2
        out = torch.cat((x, out), dim = 1)
        return out



class TemporalConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_step):
        super(TemporalConvBlock, self).__init__()
        layer = []

        self.num_layer = (np.log2(time_step)+(1.0-0.0001)).astype('int')
        for i in range(self.num_layer):
            dilation = 2**i
            layer +=[DenseBlock(in_channel+(i*out_channel), out_channel,dilation)]
        
        self.network = nn.Sequential(*layer)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.network(x)
        return x.permute(0,2,1)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, x):
        # input shape = (batch_size , channels , time_step)
        mask = np.array([[1 if i>j else 0 for i in range(x.shape[1])] for j in range(x.shape[1])])
        mask = torch.ByteTensor(mask).bool()

        #import pdb; pdb.set_trace()
        keys = self.linear_keys(x) # shape: (N, T, key_size)
        query = self.linear_query(x) # shape: (N, T, key_size)
        values = self.linear_values(x) # shape: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))
        temp = F.softmax(temp / self.sqrt_key_size, dim=1) # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = torch.bmm(temp, values) # shape: (N, T, value_size)
        return torch.cat((x, temp), dim=2) # shape: (N, T, in_channels + value_size)


class Snail(nn.Module):
    def __init__(self ,N, K, in_channels, out_channels, key_size, value_size):
        super(Snail, self).__init__()
        '''
        x shape: 1488, 19, 22, 22
        y shape: 1488, 2
        
        '''

        self.N = N
        self.K = K
        self.channel    = out_channels
        self.key_size   = key_size
        self.value_size = value_size
    
        #(channels, key_size, value_size)
        self.num_layer  = (np.log2(self.N * self.K +1)+(1.0-0.0001)).astype('int')
        

        #---------  embed  ----------
        self.embed      = EmbedNet(in_channels, out_channels)
        self.channel    += self.N

        # --------- attentation ----------
        self.attention1 = AttentionBlock(self.channel, self.key_size, self.value_size)
        self.channel    = self.channel + self.value_size
        #---------- tcn ------------
        self.tc1        = TemporalConvBlock(self.channel, 128, self.N * self.K +1)
        self.channel    += 128*self.num_layer
        self.key_size   *= 2
        self.value_size *= 2
        

        # ---------- attentation -------------
        self.attention2 = AttentionBlock(self.channel, self.key_size, self.value_size)
        self.channel    = self.channel + self.value_size
        
        # ----------- tcn ----------
        self.tc2        = TemporalConvBlock(self.channel, 128, self.N * self.K +1)
        self.channel    += 128*self.num_layer
        self.key_size   *= 2
        self.value_size *= 2
        
        # ----------- attentation -----------
        self.attention3 = AttentionBlock(self.channel, self.key_size, self.value_size)
        self.channel    = self.channel + self.value_size
        self.fc         = nn.Linear(self.channel, self.N)
        self.softmax    = nn.Softmax(dim = 1)
    

    def forward(self, x, y):
        x    = self.embed(x.float())
        x = torch.cat((x, y), 1)

        x = x.reshape(-1, self.N * self.K + 1, x.shape[1])
        x = self.attention1(x)
        x = self.tc1(x)
        x = self.attention2(x)
        x = self.tc2(x)
        x = self.attention3(x)
        
        out = self.fc(x)
        
        out = self.softmax(out[:, -1, :])
        return out


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1)
        self.bn    = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)
        return out

class EmbedNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, in_channels=19, out_channels = 64):
        super(EmbedNet, self).__init__()
        self.h1 = ResBlock(in_channels = in_channels, out_channels = out_channels)
        self.h2 = ResBlock(in_channels = out_channels, out_channels = out_channels)
        self.h3 = ResBlock(in_channels = out_channels, out_channels = out_channels)
        self.h4 = ResBlock(in_channels = out_channels, out_channels = out_channels)
        
    def forward(self, x):
        # print(x.shape)
        x = self.h1(x)
        # print(x.shape)
        x = self.h2(x)
        # print(x.shape)
        x = self.h3(x)
        # print(x.shape)
        x = self.h4(x)
        # print(x.shape)
        

        return x.view(x.size(0), -1)



if __name__ == "__main__":
    model = EmbedNet()
    data = torch.randn(31710, 19, 22, 22)
    _ = model(data)

        
            

