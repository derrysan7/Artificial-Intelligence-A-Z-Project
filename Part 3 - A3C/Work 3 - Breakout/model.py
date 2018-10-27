# AI for Breakout

# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initializing and setting the variances of a tensor of weights
def normalized_columns_initializer(weights, std = 1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out)) # pow = squared it, sum(1) = sum with respect to dimension 1, expand_out will get the weights of out # var(out) = std^2
    return out

# Initializing the weights of the neural network for an optimal learning
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4]) #dim1 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4])*weight_shape[0] #dim0 * dim2 * dim3
        w_bound = np.sqrt(6. / fan_in + fan_out) #the size of the tensor of weights
        m.weight.data.uniform_(-w_bound, w_bound) #generate random weight that inversely proporsional to the size of the tensor of weight
        m.bias.data.fill_(0) #the bias
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / fan_in + fan_out)
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        
# Making the A3C brain

class ActorCritic(torch.nn.Module):
    
    def __init__(self, num_inputs, action_space): #num_inputs = dimension of our input images, action_space = the space that contain the action and number of action
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256) # with lstm we can learn some long temporal relationship, t+1 will depend on t, t-1, t-2,..., t-n #256 = number of output neurons of the lstm
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1) #output = V(S)
        self.actor_linear = nn.Linear(256, num_outputs) #output = Q(S,A)
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train() #a method to put the module in train mode. it allows to activate if there is any dropout in the batch normalization
        
    def forward(self, inputs): #inputs = input images, hidden node and cell node of the lstm
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs)) #Exponensial Linear Unit
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32 * 3 * 3) #-1 = one dimensional vector
        (hx, cx) = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
    
        
        
        
    












