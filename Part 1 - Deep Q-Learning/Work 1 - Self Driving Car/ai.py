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
        samples = zip(*random.sample(self.memory, batch_size)) #if list = ((1,2,3),(4,5,6)) ; zip(*list) = ((1,4),(2,3),(5,6)) ; (state,action,reward) #we'll wrapt the batches into pytorch variable (a variable that contain both a tensor and a gradient)
        return map(lambda x: Variable(torch.cat(x, 0)), samples) #this lambda function will take the samples concatenate them with respect to the 1st dimension, and then eventually we convert this tensors into some torch variables that contain both a tensor and a gradient. so that later when we applied sgd, we'll be able to differentitate to update the weights

#Implementing Deep Q Learning
        
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma #gamma = delayed coefficient
        self.reward_window = [] #collection of mean of last 100 reward
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) #gradient descent, lr(learning rate)
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #input_size 5 dimension : 3 sensor, orientation, -orientation ; 0 because the network expected  the fake dimension has to be the first dimension
        self.last_action = 0 #index 0=0, 1=20, 2=-20
        self.last_reward = 0
        
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*7) #we won't be including the gradient with the input state using volatile true to save some memory ; T=7 (temporal parameter, closer to zero meaning the car will be less sure to take that action)
        action = probs.multinomial() #will give us random draw from the distribution probs
        return action.data[0,0]
        
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action): #need batch_state and batch_next_state to compute the loss
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) #predictions
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target) #td loss will be equal to huber loss
        self.optimizer.zero_grad() #zero_grad will reinitialize the optimizer at each iteration of the sgd loop
        td_loss.backward(retain_variables = True) #to perform backpropagation back into the network; retain_variables is to free the memory
        self.optimizer.step() #using the optimizer to update the weight
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0) #the 5 signal is the state
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100) #get 100 random batches from memory
            self.learn(batch_state, batch_next_state, batch_reward, batch_action) #the learning happen
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000: #the last 1000 means of the last 100 reward
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")




           
            
            



            