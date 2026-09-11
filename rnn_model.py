import torch 
import torch.nn as nn


# many to one RNN architecture
class RNN(nn.Module):
    def __init__(self,input_size, hidden_size=128, num_layer=1):  # 1RNN = 1  num_layer input size=40k
        super().__init__()

        self.hidden_size=hidden_size
        self.num_layers=num_layer
        
        self.rnn=nn.RNN(input_size , hidden_size, num_layer,batch_first=True )  # batch_first changing the shape of the input becz we avoide to do reshape operation

        #fully connected layer 
        self.fc=nn.Linear(hidden_size,1) # hidden layer size and op 1 require
        
    def forward(self,x):
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size) # x.size is batch size 
        output,_=self.rnn(x,h0)
        #1st value= hidden state of all the timestamp=(batch,seq_len, hidden size)
        #2nd value=final hidden layer of last timestamp

        output=self.fc(output[:,-1,:]) # all batch size -1 means last timestamp length and all hidden layer size
        return output