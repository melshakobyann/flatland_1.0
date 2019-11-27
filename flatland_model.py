import torch
import torch.nn as nn
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')




class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.advantages = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.advantages[:]




class Decoder(nn.Module):
  def __init__(self, input_size, hidden_size, action_size, num_layers, bidirectional = True):
    super(Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.bidirectional = bidirectional
    self.num_layers = num_layers
    
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
    self.actor_head_linear = nn.Linear(hidden_size*2, action_size)
    self.actor_head_final = nn.Softmax(dim=-1)
    self.critic_head = nn.Linear(hidden_size*2, 1)
  
  def forward(self, inputs, hidden):

    output, hidden = self.lstm(inputs, hidden)
    value = self.critic_head(output)
    policy = self.actor_head_linear(output)
    policy = self.actor_head_final(policy)
    return value, policy
    
  def init_hidden(self, batch_size):
    weight = next(self.parameters()).data
    return (Variable(weight.new(self.num_layers * (1 + int(self.bidirectional)), batch_size, self.hidden_size).zero_()),
           Variable(weight.new(self.num_layers * (1 + int(self.bidirectional)), batch_size, self.hidden_size).zero_()))



# x = Decoder(40, 6, 4)
# y, v = x.forward(a, x.init_hidden()) #Assuming <SOS> to be all zeros
# print(y.shape)
# # print(v[0].shape)
# # print(v[1].shape)
# print(y)
# print(v)