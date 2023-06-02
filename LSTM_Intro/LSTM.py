import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import lightning as L

class myLSTM(L.LightningModule):
    def __init__(self):
        super().__init__()
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        # LSTM parameters for long-term memory
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.normal(mean=torch.tensor(0.0)), requires_grad=True)

        # LSTM parameters for potential memory
        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.normal(mean=torch.tensor(0.0)), requires_grad=True)
        
        # LSTM parameters for potential memory
        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.normal(mean=torch.tensor(0.0)), requires_grad=True)
      
        # LSTM parameters for output memory
        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.normal(mean=torch.tensor(0.0)), requires_grad=True)
        
    def lstm_unit(self, input_value, long_memory, short_memory):
        # Calculate the long-term memory gate
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) + (input_value * self.wlr2) + self.blr1)
        
        # Calculate the potential memory gate
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) + (input_value * self.wpr2) + self.bpr1)
        
        # Calculate the potential memory
        potential_memory = torch.tanh((short_memory * self.wp1) + (input_value * self.wp2) + self.bp1)
        
        # Update the long-term memory
        updated_long_term_memory = (long_memory * long_remember_percent) + (potential_memory * potential_remember_percent)
        
        # Calculate the output gate
        output_percent = torch.sigmoid((short_memory * self.wo1) + (input_value * self.wo2) + self.bo1)
        
        # Update the short-term memory
        updated_short_term_memory = torch.tanh(updated_long_term_memory) * output_percent

        return [updated_long_term_memory, updated_short_term_memory]

    def forward(self, input):
        long_memory = 0
        short_memory = 0
        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        # Apply LSTM units iteratively to update the memories
        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day3, long_memory
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)
        
        return short_memory

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = (output_i - label_i) ** 2
        self.log('train_loss', loss)  # Using Lightning module to keep track
        if label_i == 0:
            self.log('out_0', output_i)
        else:
            self.log('out_1', output_i)
        return loss

model = myLSTM()
print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1])).detach())
