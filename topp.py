import numpy as np
from visualization import HexMapVisualizer
from AI import NeuralNetActor
from hexgame import HexGame
from utils import max_index
from copy import deepcopy
from random import random, choice


class TOPP:
    def __init__(self, M, 
                    G, 
                    E, 
                    size, 
                    nn_layers, 
                    lr, 
                    activation, 
                    optimizer, 
                    loss_func, 
                    save_folder,
                    display=False,
                    delay=1):
        self.vs = [(a, b) for a in range(M) for b in range(M) if a < b]
        self.model_postfixes = [int(E/(M-1)) * i for i in range(M)]
        self.number_of_games = 2 if display else G
        self.size = size
        self.layers = nn_layers
        self.lr = lr
        self.activation = activation
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.save_folder = save_folder
        self.display = display
        self.delay = delay
        self.AI = {}

    def init_AI(self):
        for i, pf in enumerate(self.model_postfixes):
            ai = ToppPlayer(NeuralNetActor(self.size**2, self.layers, self.lr, self.activation, self.optimizer, self.loss_func, self.save_folder), pf, i)
            ai.load(self.save_folder)
            self.AI[i] = ai

    def change_turn(self, players, player):
        if player.index == players[0]:
            r = players[1]
        else:
            r = players[0]

        return self.AI[r]

    def tournament(self):

        self.init_AI()
        
        for players in self.vs:
            print(f"Model {self.model_postfixes[players[0]]} vs. model {self.model_postfixes[players[1]]}")
            starter = 1
            game = HexGame(self.size, starter)
            visualizer = HexMapVisualizer(game.board.cells.values(), True, self.size, game_type="hex")

            smarter_wins = 0
            
            for i in range(self.number_of_games):
                
                player = self.AI[players[i % 2]]
                strtr = player.name
                #print(f"Player {game.playing} (model {player.name}) is starting!")


                while not game.is_terminal_state():
                    #print(f"Player {player.name} moving")
                    if self.display: visualizer.draw(game.get_state(), self.delay)

                    state = game.get_simple_state()
                    legal_moves = game.get_reversed_binary()

                    possible_states = game.generate_possible_child_states()

                    pred, idx = player.model.get_move(state, legal_moves)
                    
                    if random() > (0 if self.display else 0.2): 
                        best_index = idx
                    else:
                        best_index = np.random.choice(np.arange(len(pred)), p=pred)
                    

                    data = possible_states[best_index]

                    game.do_action(data["action"])
                    
                    prev_player = player.name
                    player = self.change_turn(players, player)
                

                if self.display: visualizer.draw(game.get_state(), self.delay*3)
                #print(f"Model {strtr} started, and model {prev_player} (player {game.playing}) won!")

                smarter = self.model_postfixes[players[1]]
                if prev_player == smarter:
                    smarter_wins += 1


                #starter = 2 if starter == 1 else 1

                game.reset(starter, hard=True)
            
            print(f"Model {smarter} won {smarter_wins} out of {self.number_of_games} games ({smarter_wins/self.number_of_games*100}%)")


class ToppPlayer:
    def __init__(self, model, name, index):
        self.model = model
        self.name = name
        self.index = index

    def load(self, folder):
        print(self.name)
        self.model.load_model(f"{folder}/checkpoint{self.name}.pth.tar")


# Neural net
from torch.optim import Adagrad, SGD, RMSprop, Adam
from torch.nn import ReLU, Sigmoid, Tanh, Linear, BCELoss, LeakyReLU
from math import sqrt

if __name__ == "__main__":
    SIZE                    = 3
    EPISODES                = 250
    M                       = 4
    G                       = 24
    NN_LEANRING_RATE        = 0.0005
    NN_HIDDEN_LAYERS        = [256, 256, 128, 128]
    NN_ACTIVATION           = ReLU
    NN_OPTIMIZER            = Adam
    NN_LOSS_FUNCTION        = BCELoss
    EPSILON                 = 1
    EPSILON_DR              = 0.99
    MC_EXPLORATION_CONSTANT = sqrt(2)
    MC_NUMBER_SEARCH_GAMES  = 50
    MIXED_START = False
    SAVE_FOLDER = "models_3_200"

    topp = TOPP(M, 
                G, 
                EPISODES, 
                SIZE, 
                NN_HIDDEN_LAYERS, 
                NN_LEANRING_RATE,
                NN_ACTIVATION, 
                NN_OPTIMIZER, 
                NN_LOSS_FUNCTION,
                SAVE_FOLDER,
                display=False)

    topp.tournament()

                    
