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
        self.padding = (kernal_size-1)*dilation
        self.conv    = nn.Conv1d(in_channels, out_channels, kernal_size, 
                                padding = self.padding, stride = 1, dilation = dilation)
    def forward(self, x):
        # input shape (batch_size , channel , time_step)
        out = self.conv(x)
        return out[:, :, :-1*self.padding]




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
            if(i == 0):
                layer +=[DenseBlock(in_channel, out_channel,dilation)]
            else:
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
        mask = np.array([[1 if i>j else 0 for i in range(input.shape[1])] for j in range(input.shape[1])])
        mask = torch.ByteTensor(mask).cuda()

        #import pdb; pdb.set_trace()
        keys = self.linear_keys(input) # shape: (N, T, key_size)
        query = self.linear_query(input) # shape: (N, T, key_size)
        values = self.linear_values(input) # shape: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))
        temp = F.softmax(temp / self.sqrt_key_size, dim=1) # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = torch.bmm(temp, values) # shape: (N, T, value_size)
        return torch.cat((input, temp), dim=2) # shape: (N, T, in_channels + value_size)


class Snail(nn.Module):
    def __init__(self ,N, K, in_channel, batch_size):
        super(Snail, self).__init__()
        self.N = N
        self.K = K
        self.channel    = in_channel
        self.key_size   = 64
        self.value_size = 32
        self.time_step  = N * K + 1
        #(channels, key_size, value_size)
        self.num_layer = (np.log2(self.time_step)+(1.0-0.0001)).astype('int')
        
        self.embed       = EmbedNet(in_channel-2, in_channel-2)
        
        self.attention1 = AttentionBlock(self.channel, self.key_size, self.value_size)
        self.channel    = self.channel + self.value_size
        
        self.tc1        = TemporalConvBlock(self.channel, 128, self.time_step)
        self.channel    += 128*self.num_layer
        self.key_size   *= 2
        self.value_size *= 2
        
        self.attention2 = AttentionBlock(self.channel, self.key_size, self.value_size)
        self.channel    = self.channel + self.value_size
        
        self.tc2        = TemporalConvBlock(self.channel, 128, self.time_step)
        self.channel    += 128*self.num_layer
        self.key_size   *= 2
        self.value_size *= 2
        
        self.attention3 = AttentionBlock(self.channel, self.key_size, self.value_size)
        self.channel    = self.channel + self.value_size
        self.fc         = nn.Linear(self.channel, 2)
        self.softmax    = nn.Softmax(dim = 1)
    

    def forward(self,x,y):
        last_index = []
        batch = self.N*self.K + 1 
        x    = self.embed(x.float())

        for i in range(len(y)):
            if((i+1) % batch == 0):
                last_index.append(i)
        
        x = add_one_hot(x, y, last_index)
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

    def __init__(self, in_channels, out_channels, pool_padding=0):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size= 3) 
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size= 3) 
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size= 3) 
        self.bn3   = nn.BatchNorm2d(filters)
        self.relu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size= 3) 

        self.maxpool = nn.MaxPool2d(2, padding=pool_padding)
        self.dropout = nn.Dropout(p=0.9)

    def forward(self, x):
        residual = self.conv4(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out += residual
        out = self.maxpool(out)
        out = self.dropout(out)

        return out

class EmbNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, in_channels=3):
        super(EmbNet, self).__init__()
        self.block1 = ResBlock(in_channels, 64)
        self.block2 = ResBlock(64, 96)
        self.block3 = ResBlock(96, 128, pool_padding=1)
        self.block4 = ResBlock(128, 256, pool_padding=1)
        self.conv1 = conv(256, 2048, kernel_size=1, padding=0)
        self.maxpool = nn.MaxPool2d(6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.9)
        self.conv2 = conv(2048, 384, kernel_size=1, padding=0)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x.view(x.size(0), -1)

        
            

