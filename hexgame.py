from hexmap import DiamondHexMap
from mcts import MCTS, Node
from visualization import HexMapVisualizer
from AI import NeuralNetActor, ReplayBuffer
from matplotlib import pyplot as plt

from random import choice, sample
from tqdm import tqdm
from math import sqrt
from copy import copy

#Profiling
import cProfile, pstats, io
from pstats import SortKey

class HexGame:
    """ 
        The State Manager 
    """
    def __init__(self, size, playing):
        self.size = size
        self.playing = playing
        self.board = DiamondHexMap(size, True, size**2)
        self.board_copy = None
        self.board.init_map()
        
    def get_state(self):
        return [(x.pos, x.owned) for x in self.board.cells.values()]

    def do_action(self, pos):
        cell = self.board.cells[pos]
        cell.set_peg(self.playing)
        self.playing = self.next_player()

    def generate_possible_child_states(self):
        """
            Actions represent the states they lead to.
        """
        child_actions = []
        p = self.playing
        p_next = self.next_player()
        state = self.get_state()
        for cell in state:
            pos, owner = cell
            if owner == 0:
                state_ = [p if pos_ == pos else owned for pos_, owned in state]
                reverse = self.get_reversed_binary(s=state_)
                state_.append(p_next)
                child_actions.append({"state": state_, "player": p_next, "action": pos, "reverse": reverse})
        return child_actions

    def is_terminal_state(self):
        p = self.next_player()
        check_worthy = [cell for cell in self.board.cells.values() if cell.pos[p - 1] == self.size - 1]
        for c in check_worthy:
            if c.owned == p and c.connected:
                self.playing = self.next_player()
                return True
        return False

    def next_player(self):
        return 2 if self.playing == 1 else 1

    def get_simple_state(self):
        s = [x.owned for x in self.board.cells.values()]
        s.append(self.playing)
        return s

    def get_reversed_binary(self, s=None):
        """ Reverse state to highlight possible actions """
        if s == None:
            r = [0 if x.owned == 1 or x.owned == 2 else 1 for x in self.board.cells.values()]
        else:
            r = [0 if x == 1 or x == 2 else 1 for x in s]
        return r

    def copy_board(self):
        self.board_copy = [(x.owned, x.connected) for x in self.board.cells.values()]

    def reset_map(self, playing, hard=False):
        self.playing = playing
        if hard:
            self.board.reset_map()
        else:
            self.board.set_map(self.board_copy)


if __name__ == "__main__":
    EPISODES = 450
    M = 4
    NN_LEANRING_RATE = 0.1
    NN_HIDDEN_LAYERS = [32, 64, 128, 256, 128, 64, 32]
    NN_ACTIVATION = "RELU HER"
    NN_OPTIMIZER = "ADAM HER"
    NN_LOSS_FUNCTION = "BCELoss HER"
    EPSILON = 0.99
    EPSILON_DR = 0.99
    MC_EXPLORATION_CONSTANT = sqrt(2)
    MC_NUMBER_SEARCH_GAMES = 500


    size = 5
    player = 1
    actual_player = 1
    mc_player = 1

    i_save = None
    RBUF = ReplayBuffer()
    ANET = NeuralNetActor(size**2, NN_HIDDEN_LAYERS, NN_LEANRING_RATE)

    pr = cProfile.Profile()

    # Stateful game used for actual game
    actual_game = HexGame(size, player)
    visualizer = HexMapVisualizer(actual_game.board.cells.values(), True, size, game_type="hex")
    
    mc_game = HexGame(size, player)
    mc = MCTS(mc_game, MC_EXPLORATION_CONSTANT, a_net=ANET, epsilon=EPSILON)
    
    pr.enable()
    for i in tqdm(range(EPISODES + 1)):
        if i % int(EPISODES/(M-1)) == 0.0:
            ANET.save_model(str(i))

        # No action needed to reach initial state
        action = None

        state = mc_game.get_simple_state()

        # Init Monte Carlo game board
        root = Node(state, player, None, action, mc_game.get_reversed_binary())
        
        move_count = 0
        while not actual_game.is_terminal_state():
            #visualizer.draw(actual_game.get_state(), 0.000001)

            # Find the best move using MCTS
            new_root, prev_root_children = mc.tree_search(root, MC_NUMBER_SEARCH_GAMES)
            
            # Distribution of visit counts along all arcs emanating from root
            D = [child.visits for child in prev_root_children]

            # Add case to RBUF
            RBUF.add(root.state, root.reversed_state, D)

            root = new_root

            action = root.action

            actual_game.do_action(action)
            mc.game.do_action(action)
            
            root.reset()


        print("Player {} won!".format(actual_game.playing))
        #visualizer.draw(actual_game.get_state(), 0.000001)

        state_batch, legal_moves, d_batch = RBUF.get_sample()

        ANET.train(state_batch, legal_moves, d_batch)

        # Mix starting players
        player = 2 if player == 1 else 1

        actual_game.reset_map(player, hard=True)
        mc_game.reset_map(player, hard=True)
        EPSILON *= EPSILON_DR    
    pr.disable()

    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open('test.txt', 'w+') as f:
        f.write(s.getvalue())

    plt.clf()
    d = ANET.losses
    plt.plot(list(d.keys()), list(d.values()))
    plt.show()


    """
    game = HexGame(size)
    visualizer = HexMapVisualizer(game.board.cells.values(), True, size, game_type="hex")

    g = True
    while g:
        curr_state = game.get_state()
        visualizer.draw(curr_state, 0.0000001)

        actions = game.generate_possible_actions(curr_state)
        input(actions)
        c = choice(actions)
        game.do_action(c, player)

        if game.is_terminal_state(player):
            print("Player {} won!".format(player))
            visualizer.draw(game.get_state(), 0.0000001)
            player = 1
            game = HexGame(size)
            visualizer = HexMapVisualizer(game.board.cells.values(), True, size, game_type="hex")

        player = 2 if player == 1 else 1

    print("Player {} won!".format(player))
    visualizer.draw(game.get_state(), 1000)
    """

