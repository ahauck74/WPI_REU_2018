import torch, torch.nn as nn
import numpy as np
import math
from copy import copy


###########################################################
#Models
###########################################################        
        
class MyRNN(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(MyRNN, self).__init__()
        self.rnn = torch.nn.RNN(input_dim, hidden_size)
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(hidden_size, 2)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)

    def forward(self, input_sequence, hidden):
        output, hidden = self.rnn(input_sequence, hidden)
        prediction = self.layer1(output[-1])
        prediction = self.soft(prediction)
        return(prediction, hidden)

class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(MyLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_size)
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(hidden_size, 2)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)

    def forward(self, input_sequence):
        output, hidden = self.lstm(input_sequence, hidden)
        prediction = self.layer1(output[-1])
        prediction = self.soft(prediction)
        return(prediction, hidden)
    
class MyGRU(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(MyGRU, self).__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_size)
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(hidden_size, 2)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)

    def forward(self, input_sequence, hidden):
        output, hidden = self.gru(input_sequence, hidden)
        prediction = self.layer1(output[-1])
        prediction = self.soft(prediction)
        return(prediction, hidden)
    
class ManualGRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(ManualGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self._steps = seq
        self.output_dim = output_dim
        self.combined_dim = hidden_dim + input_dim
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.soft = nn.Softmax(dim=1)
        self.z = nn.Linear(self.combined_dim, self.hidden_dim) #Update gate
        self. r = nn.Linear(self.combined_dim, self.hidden_dim) #Reset gate
        self.h = nn.Linear(self.combined_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward_step(self, x, h):
        s = copy(h)
        combined = torch.cat((x, s), dim=2)                                                                                              
        z = self.sig(self.z(combined))
        r = self.sig(self.r(combined))
        h = torch.cat((x, torch.mul(r, s)), dim=2)
        h = self.tanh(self.h(torch.cat((x, torch.mul(s, r)), dim=2)))                                     
        hidden = ((1-z)*h) + (z*s) # output new hidden                                                                   
        return hidden
    
    def forward(self, input, hidden):
        if self._steps == 0: self._steps = len(input)                                                                                                  
        for i in range(self._steps):
            x = input[i].unsqueeze(0)  
            hidden = self.forward_step(x, hidden)                                                                                                     
        output = self.out(hidden)
        print('output', output)
        prediction = self.soft(output[0])                                                                                        
        return prediction, hidden
    
    
class GRU_D(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(GRU_D, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = int(input_dim/3)
        self._steps = seq
        self.output_dim = output_dim
        self.combined_dim = hidden_dim + 2*self.input_dim
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.soft = nn.Softmax(dim=1)
        self.gamma = nn.Linear(1, 1)
        self.z = nn.Linear(self.combined_dim, self.hidden_dim) #Update gate
        self. r = nn.Linear(self.combined_dim, self.hidden_dim) #Reset gate
        self.h = nn.Linear(self.combined_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        
    def set_mean(self, mean):
        self.mean = torch.FloatTensor(mean)
    
    def forward_step(self, x, h, m, dt):
        gamma = torch.zeros(1,dt.shape[1], 1)
        
            
        for i, val in enumerate(dt[0,:,0]): #Loop through the batch
            gamma[0,i,0] = math.exp(-max(0, self.gamma(dt[:,i,:])))
            if not np.isnan(x[0,i,0]): #If not missing, update the most recent value
                self.xprime[i] = x[0,i,0]
            
            #Impute x if missing
            else:
                x[0,i,0] = (1-m[0,i,0])*(gamma[0,i,0]*self.xprime[i] + torch.mul(self.mean, (1-gamma[0,i,0])))#Is the mean changing? For now just trying to return mean from imputation(), calculating the mean every time in forward or forward step might be computationally intensive
        
        s = copy(h) 
        s = torch.mul(gamma, s)
        combined = torch.cat((x, s), dim=2)       
        combined = torch.cat((combined, m), dim=2) 
        z = self.sig(self.z(combined))
        r = self.sig(self.r(combined))
        h = torch.cat((x, torch.mul(r, s)), dim=2)#Does order of concatenation matter?
        h = torch.cat((h, m), dim=2)
        h = self.h(h)
        h = self.tanh(h)
        hidden = ((1-z)*h) + (z*s) # output new hidden                                                                   
        return hidden
    
    def forward(self, input, hidden):
        self.xprime = [0]*(input.shape[1]+1)
        if self._steps == 0: self._steps = len(input)                                                                                                  
        for i in range(self._steps): #Loop through the time series
            x = input[i, :, 0].unsqueeze(-1).unsqueeze(0)
            m = input[i, :, 1].unsqueeze(-1).unsqueeze(0)
            dt = input[i, :, 2].unsqueeze(-1).unsqueeze(0)
            
            hidden = self.forward_step(x, hidden, m, dt)                                                                                                     
        output = self.out(hidden)
        prediction = self.soft(output[0])                                                                                        
        return prediction, hidden
