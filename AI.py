import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.autograd import Variable

from collections import defaultdict
from random import randrange, sample
from abc import ABC, abstractmethod

class RLearner:
    """
        The agent that owns the actor and critic
    """
    def __init__(self, 
                actor):
        self.actor = actor
        
    def decide_action_to_take(self, state, epsilon, possible_actions):
        """
        Decides whether to do an action based on policy or randomly.
        If the state is unknown, the choice will always be random.
        """
        if randrange(10000)/10000 < epsilon:
            action = self.get_random_action(possible_actions)
        else:
            if state in (x[0] for x in self.actor.policy.keys()):
                action = self.actor.select_action(state)
            else:
                action = self.get_random_action(possible_actions)

        return action

    def get_random_action(self, possible_actions):
        action = None

        if len(possible_actions) > 0:
            random_index = randrange(len(possible_actions))
            action = possible_actions[random_index]
        return action


class Actor:
    def __init__(self, learning_rate, discount_factor, e_decay_rate):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.e_decay_rate = e_decay_rate


class NeuralNetActor:
    def __init__(self, input_size, layers, learning_rate, activation, optimizer, loss_func, save_folder, epochs=10):
        self.input_size = input_size
        self.activation = activation()
        self.model = self.init_neural_net(layers)
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.criterion = loss_func(reduction="mean")
        self.save_folder = save_folder
        self.epochs = epochs
        self.global_step = 0
        self.losses = {}
        print(self.model)

    def init_neural_net(self, layers):

        container_modules = []

        # Input layer
        container_modules.append(nn.Linear(self.input_size + 1, layers[0]))
        container_modules.append(self.activation)

        # Hidden layer(s)
        for i in range(len(layers)-1):
            container_modules.append(nn.Linear(layers[i], layers[i+1]))
            container_modules.append(self.activation)

        # Output layer
        container_modules.append(nn.Linear(layers[-1], self.input_size))
        container_modules.append(nn.Softmax(-1))

        return nn.Sequential(*container_modules)

    def train(self, states, legal_moves, dists):
        pred = self.forward(states)
        target = torch.FloatTensor(dists) #torch.tensor(dists, dtype=torch.float)

        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses[self.global_step] = loss
        self.global_step += 1
        
    def forward(self, states):
        s_tensor = torch.FloatTensor(states)
        return self.model(s_tensor)

    def get_move(self, state, lm):
        """
            Produces probability distribution over all k^2 (k = size of board) 
            moves, and rescales them to a distribution over only the legal 
            k^2 - q moves (where q is the moves no longer available)
        """
        s_tensor = torch.FloatTensor(state)
        length = len([m for m in lm if m == 1])
        pred = self.forward(s_tensor).data.numpy()
        sum_ = 0
        new_pred = np.zeros(length)
        npi = 0
        for i in range(self.input_size):
            if lm[i]:
                sum_ += pred[i]
                new_pred[npi] = pred[i]
                npi += 1
        new_pred /= sum_
        greedy_index = np.argmax(new_pred)
        return new_pred, greedy_index

    def save_model(self, num):
        torch.save(self.model.state_dict(), f"{self.save_folder}/checkpoint{num}.pth.tar")

    def load_model(self, PATH = "models/checkpoint0.pth.tar"):
        """
            Loads the model weights from file into the model
        """
        print(PATH)
        try:
            self.model.load_state_dict(torch.load(PATH))
            self.model.eval()
        except:
            print(PATH)
            raise AttributeError

def rescale(state, distribution):
    length = len(state) - 1
    r = np.zeros(length)
    for i in range(length):
        if state[i] == 0:
            r[i] = distribution.pop(0)
    return r

def normalize(a):
    return a / np.linalg.norm(a, ord=1)

class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        self.batch_size = 250

    def add(self, state, reverse, D):
        """
        if sum(D) > 0:
            D = normalize(D)
        """
        D = rescale(state, D)
        self.buffer.append((state, reverse, D))
        if len(self.buffer) > 500:
            self.buffer.pop(0)

    def get_sample(self):
        return map(list, zip(*sample(self.buffer, min(len(self.buffer), self.batch_size))))

    def clear(self):
        self.buffer = []


class TableActor(Actor):
    def __init__(self, learning_rate, discount_factor, e_decay_rate):
        super().__init__(learning_rate, discount_factor, e_decay_rate)
        self.policy = defaultdict(lambda: 0)
        self.e_trace = defaultdict(lambda: 0)
    
    def set_policy(self, sap, delta):
        self.policy[sap] = self.policy[sap] + self.learning_rate * delta * self.e_trace[sap]

    def set_eligibility(self, sap, reset=0):
        new_eligibility = self.discount_factor * self.e_decay_rate * self.e_trace[sap]
        self.e_trace[sap] = new_eligibility if reset == 0 else 1

    def select_action(self, state):
        """ 
            Selects the action with highest value from policy
        """
        max = float('-inf')
        action = None
        for s in (x for x in self.policy.keys() if x[0] == state):
            val = self.policy[s]
            if val >= max:
                max = val
                action = s[1]
        return action


class Critic:
    def __init__(self, e_trace, environment, learning_rate, discount_factor, e_decay_rate):
        self.e_trace = e_trace
        self.environment = environment
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.e_decay_rate = e_decay_rate
        self.TD_error = None

    def reset_e_trace(self):
        pass

    def set_eligibility(self, state, reset=0):
        pass


class TableCritic(Critic):
    def __init__(self, environment, learning_rate, discount_factor, e_decay_rate):
        super().__init__(defaultdict(lambda: 0),
                        environment,
                        learning_rate,
                        discount_factor,
                        e_decay_rate)
        self.values = defaultdict(lambda: randrange(100)/100)

    def set_value(self, state):
        self.values[state] = self.values[state] + self.learning_rate * self.TD_error * self.e_trace[state]

    def set_eligibility(self, state, reset=0):
        new_eligibility = self.discount_factor * self.e_decay_rate * self.e_trace[state]
        self.e_trace[state] = new_eligibility if reset == 0 else 1

    def set_TD_error(self, r, state, prevstate):
        self.TD_error = r + self.discount_factor * self.values[state] - self.values[prevstate]


class NeuralNetCritic(Critic):
    def __init__(self, environment, learning_rate, discount_factor, e_decay_rate, layers, input_size):
        super().__init__([],
                        environment, 
                        learning_rate, 
                        discount_factor, 
                        e_decay_rate)
        self.input_size = input_size
        self.activation = nn.ReLU()
        self.model = self.init_neural_net(layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def init_neural_net(self, layers):

        container_modules = []

        # Input layer
        container_modules.append(nn.Linear(self.input_size, layers[0]))
        #container_modules.append(self.activation)

        # Hidden layer(s)
        for i in range(len(layers)-1):
            container_modules.append(self.activation)
            container_modules.append(nn.Linear(layers[i], layers[i+1]))

        # Output layer
        container_modules.append(nn.Sigmoid())

        return nn.Sequential(*container_modules)

    def loss(self, delta):
        """
        Mean squared error of the difference between our neural network’s 
        output for the next state and the output for the current state
        """
        return torch.mean(delta**2)

    def get_value(self, state):
        """
            Performs forward pass (prediction) on state tensor
        """
        return self.model(torch.tensor(state, dtype=torch.float))

    def set_value(self, state):
        
        # Reset gradients 
        self.optimizer.zero_grad()
        
        # Compute gradients of loss-function (NOT USED)
        #self.loss(self.TD_error).backward(retain_graph=True) # Retain graph so buffers can be freed
    
        # Gradient computation as presented in Sutton & Barto (p. 232)
        self.get_value(state).backward()
        
        # Update weights for each layer 
        with torch.no_grad():
            for weight, eligibility in zip(self.model.parameters(), self.e_trace):
                eligibility *= self.discount_factor * self.e_decay_rate
                eligibility += weight.grad

                #weight.grad *= eligibility # Only for loss gradient
                
                weight.grad = -self.TD_error * eligibility
            self.optimizer.step() # w -= lr * w.grad
        

    def set_TD_error(self, r, state, prevstate):
        prevstate_value = self.get_value(prevstate)
        currstate_value = self.get_value(state)
        
        self.TD_error = r + self.discount_factor * currstate_value - prevstate_value

    def reset_e_trace(self):
        self.e_trace = [torch.tensor(np.zeros(params.shape), dtype=torch.float) for params in self.model.parameters()]




class Environment(ABC):
    @abstractmethod
    def produce_initial_state():
        pass

    @abstractmethod
    def generate_possible_child_states():
        pass

    @abstractmethod
    def is_terminal_state():
        pass

    @abstractmethod
    def perform_action():
        pass



