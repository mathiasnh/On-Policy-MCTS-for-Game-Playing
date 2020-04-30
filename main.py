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
    
    SIZE                    = 5
    EPISODES                = 350
    M                       = 4
    G                       = 24
    NN_LEANRING_RATE        = 0.001
    NN_HIDDEN_LAYERS        = [256, 256, 128, 128]
    NN_ACTIVATION           = ReLU
    NN_OPTIMIZER            = Adam
    NN_LOSS_FUNCTION        = BCELoss
    EPSILON                 = 1
    EPSILON_DR              = 0.99
    MC_EXPLORATION_CONSTANT = sqrt(2)
    MC_NUMBER_SEARCH_GAMES  = 2000
    
    DISPLAY_INDICES = []
    DISPLAY_DELAY = 0.5

    MIXED_START = False
    SAVE_FOLDER = "models_demo_short"
    SAVE = True

    # TOPP
    TOPP_ONLY = True
    ON_POLICY_DISPLAY = True
    TOPP_DELAY = 0.2

    if not TOPP_ONLY:
    
        player = 1

        RBUF = ReplayBuffer()
        ANET = NeuralNetActor(SIZE**2, 
                            NN_HIDDEN_LAYERS, 
                            NN_LEANRING_RATE, 
                            NN_ACTIVATION, 
                            NN_OPTIMIZER, 
                            NN_LOSS_FUNCTION,
                            SAVE_FOLDER)

        # Stateful game used for actual game
        actual_game = HexGame(SIZE, player)
        visualizer = HexMapVisualizer(actual_game.board.cells.values(), True, SIZE, game_type="hex")
        
        mc_game = HexGame(SIZE, player)
        mc = MCTS(mc_game, MC_EXPLORATION_CONSTANT, a_net=ANET, epsilon=EPSILON)
        
        for i in tqdm(range(EPISODES)):

            # No action needed to reach initial state
            action = None

            state = mc_game.get_simple_state()

            # Init Monte Carlo root
            root = Node(state, player, None, action, mc_game.get_reversed_binary())
            
            search_timer = 0
            while not actual_game.is_terminal_state():
                if i in DISPLAY_INDICES: visualizer.draw(actual_game.get_state(), DISPLAY_DELAY)

                # Find the best move using MCTS
                search_discount = (1 - (search_timer/SIZE**2))
                new_root, prev_root_children = mc.tree_search(root, int(MC_NUMBER_SEARCH_GAMES*search_discount))
                
                # Distribution of visit counts along all arcs emanating from root
                D = [child.visits/root.visits for child in prev_root_children]

                # Add case to RBUF
                RBUF.add(root.state, root.reversed_state, D)

                root = new_root

                action = root.action

                actual_game.do_action(action)
                mc.game.do_action(action)

                root.reset()

                search_timer += 1


            if i in DISPLAY_INDICES: 
                visualizer.draw(actual_game.get_state(), DISPLAY_DELAY, show=True)


            state_batch, legal_moves, d_batch = RBUF.get_sample()

            ANET.train(state_batch, legal_moves, d_batch)

            # Mix starting players
            if MIXED_START:
                player = choice([1,2])

            actual_game.reset(player, hard=True)
            mc_game.reset(player, hard=True)
            mc.epsilon *= EPSILON_DR
            #mc.epsilon = EPSILON - EPSILON*(i/EPISODES)**0.5    

            if i % int(EPISODES/(M-1)) == 0.0:
                if SAVE: ANET.save_model(str(i))

        plt.clf()
        d = ANET.losses
        plt.plot(list(d.keys()), list(d.values()))
        plt.show()
    
    print("Running TOPP!")
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
                display=ON_POLICY_DISPLAY,
                delay=TOPP_DELAY)

    topp.tournament()