# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #optimizer for stochastic gradient descent
import torch.autograd as autograd
from torch.autograd import Variable

#Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30) #5 input layer and 30 hidden neuron in the hidden layer
        self.fc2 = nn.Linear(30, nb_action) #30 hidden layer and 3 output layer
        
    def forward(self, state):
        x = F.relu(self.fc1(state)) #x = hidden neuron, relu = rectifier function
        q_values = self.fc2(x)
        return q_values
    
#Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity #maximum number of transitions we want to have in our memory of events.
        self.memory = [] #a simple list containing the last 100 events/transitions

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        #if list = ((1,2,3),(4,5,6)) ; zip(*list) = ((1,4),(2,3),(5,6)) ; (state,action,reward)
        #we'll wrapt the batches into pytorch variable (a variable that contain both a tensor and a gradient)
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples) #this lambda function will take the samples concatenate them with respect to the 1st dimension, and then eventually we convert this tensors into some torch variables that contain both a tensor and a gradient. so that later when we applied sgd, we'll be able to differentitate to update the weights






           
            
            
            