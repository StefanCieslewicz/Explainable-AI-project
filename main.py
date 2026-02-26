import snntorch as snn
import torch
import torch.nn as nn
import numpy as np

from model_eval import vanilla, tsa
import read_data


dtype = torch.float
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        batch_size = 36 # NOTE decide later on the actual size
        data_path = '...' # TODO
        
        input_neurons = 'number of channels'
        hidden_layers = 1003 #idk??!!
        outputs = 2 # two ig    
        
        self.time_steps = 256 # Hz times seconds(idk how many tho)  
        beta = 0.95
        
        self.fc1 = nn.Linear(input_neurons, hidden_layers)
        self.lif1 = snn.Leaky(beta=beta) # adjust threshold in case of poor spiking
        self.fc2 = nn.Linear
        self.lif2 = snn.Leaky(beta=beta)
        
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_record = []
        mem2_record = []
        
        for _ in range(self.time_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif1(cur2, mem2)
            spk2_record.append(spk2)
            mem2_record.append(mem2)
            
        return torch.stack(spk2_record, dim=0), torch.stack(mem2_record, dim=0)

net = Net().to(device)

class Layer():
    ...
    

def train(epochs: int):
    
    for epoch in epochs:
        pass