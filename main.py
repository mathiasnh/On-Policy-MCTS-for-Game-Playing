from hexgame import HexGame
from hexmap import DiamondHexMap
from mcts import MCTS, Node
from visualization import HexMapVisualizer
from AI import NeuralNetActor, ReplayBuffer
from topp import TOPP

from math import sqrt
from matplotlib import pyplot as plt
from tqdm import tqdm
from random import choice

# Neural net
from torch.optim import Adagrad, SGD, RMSprop, Adam
from torch.nn import ReLU, Sigmoid, Tanh, Linear, BCELoss, LeakyReLU


if __name__ == "__main__":
    
    SIZE                    = 3
    EPISODES                = 10
    ACTUAL_GAMES            = 100
    M                       = 4
    G                       = 4
    NN_LEANRING_RATE        = 0.001
    NN_HIDDEN_LAYERS        = [16, 32]
    NN_ACTIVATION           = ReLU
    NN_OPTIMIZER            = Adam
    NN_LOSS_FUNCTION        = BCELoss
    EPSILON                 = 0.9
    EPSILON_DR              = 0.99
    MC_EXPLORATION_CONSTANT = sqrt(2)
    MC_NUMBER_SEARCH_GAMES  = 100
    MIXED_START = True

    
    player = 1
    actual_player = 1
    mc_player = 1

    RBUF = ReplayBuffer()
    ANET = NeuralNetActor(SIZE**2, 
                        NN_HIDDEN_LAYERS, 
                        NN_LEANRING_RATE, 
                        NN_ACTIVATION, 
                        NN_OPTIMIZER, 
                        NN_LOSS_FUNCTION)

    # Stateful game used for actual game
    actual_game = HexGame(SIZE, player)
    visualizer = HexMapVisualizer(actual_game.board.cells.values(), True, SIZE, game_type="hex")
    
    mc_game = HexGame(SIZE, player)
    mc = MCTS(mc_game, MC_EXPLORATION_CONSTANT, a_net=ANET, epsilon=EPSILON)

    for i in tqdm(range(EPISODES)):
        print(mc.epsilon)
        
        # Save interval for ANET params
        i_s = str(i)

        # Clear Replay Buffer
        RBUF.clear()

    
        for j in range(ACTUAL_GAMES):

            # No action needed to reach initial state
            action = None

            state = mc_game.get_simple_state()

            # Init Monte Carlo root
            root = Node(state, player, None, action, mc_game.get_reversed_binary())
            
            move_count = 0
            while not actual_game.is_terminal_state():
                #visualizer.draw(actual_game.get_state(), 0.1)

                # Find the best move using MCTS
                new_root, prev_root_children = mc.tree_search(root, MC_NUMBER_SEARCH_GAMES)
                
                # Distribution of visit counts along all arcs emanating from root
                D = [child.q for child in prev_root_children]

                # Add case to RBUF
                RBUF.add(root.state, root.reversed_state, D)

                root = new_root

                action = root.action

                actual_game.do_action(action)
                mc.game.do_action(action)
                
                root.reset()


            #print("Player {} won!".format(actual_game.playing))
            #visualizer.draw(actual_game.get_state(), 1)

            state_batch, legal_moves, d_batch = RBUF.get_sample()
            #input(f"{state_batch}, {legal_moves}, {d_batch}")

            ANET.train(state_batch, legal_moves, d_batch)

            # Mix starting players
            if MIXED_START:
                player = choice([1,2])

            actual_game.reset_map(player, hard=True)
            mc_game.reset_map(player, hard=True)
        #EPSILON *= EPSILON_DR
        mc.epsilon = EPSILON - EPSILON*(i/EPISODES)**0.5    

        if i % int(EPISODES/(M-1)) == 0.0:
            ANET.save_model(str(i))

    plt.clf()
    d = ANET.losses
    plt.plot(list(d.keys()), list(d.values()))
    plt.show()
    

    topp = TOPP(M, G, EPISODES, SIZE, NN_HIDDEN_LAYERS, NN_LEANRING_RATE,NN_ACTIVATION, NN_OPTIMIZER, NN_LOSS_FUNCTION)

    topp.tournament()