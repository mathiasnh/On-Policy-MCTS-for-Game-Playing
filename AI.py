import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.tensor
from collections import defaultdict
from random import randrange
from abc import ABC, abstractmethod

class RLearner:
    """
        The agent that owns the actor and critic
    """
    def __init__(self, 
                actor, 
                critic):
        self.actor = actor
        self.critic = critic
        
    def decide_action_to_take(self, state, epsilon):
        """
        Decides whether to do an action based on policy or randomly.
        If the state is unknown, the choice will always be random.
        """
        if randrange(10000)/10000 < epsilon:
            action = self.get_random_action()
        else:
            if state in (x[0] for x in self.actor.policy.keys()):
                action = self.actor.select_action(state)
            else:
                action = self.get_random_action()

        return action

    def get_random_action(self):
        possible_actions = self.critic.environment.generate_possible_child_states()
        action = None

        if len(possible_actions) > 0:
            random_index = randrange(len(possible_actions))
            action = possible_actions[random_index]
        return action


class Actor:
    def __init__(self, learning_rate, discount_factor, e_decay_rate):
        self.policy = defaultdict(lambda: 0)
        self.e_trace = defaultdict(lambda: 0)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.e_decay_rate = e_decay_rate

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



