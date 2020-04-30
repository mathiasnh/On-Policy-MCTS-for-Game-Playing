from hexmap import DiamondHexMap

from random import choice, sample 
from tqdm import tqdm
from math import sqrt
from copy import copy


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
        return [(x.owned, x.connected) for x in self.board.cells.values()]

    def reset(self, playing, hard=False, board_copy=None):
        self.playing = playing
        if hard:
            self.board.reset_map()
        else:
            self.board.set_map(board_copy)

    def copy(self):
        return self.copy_board(), self.playing

