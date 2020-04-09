from hexmap import DiamondHexMap
from random import choice
from visualization import HexMapVisualizer

class HexGame:
    def __init__(self, size):
        self.size = size
        self.board = DiamondHexMap(size, True, size**2)
        self.board.init_map()
        
    def get_state(self):
        return [(x.pos, x.owned) for x in self.board.cells.values()]

    def place_peg(self, pos, player):
        cell = self.board.cells[pos]
        cell.set_peg(player)

    def generate_possible_actions(self, state):
        child_actions = []
        for cell in state:
            pos, owner = cell
            if owner == 0:
                child_actions.append(pos)
        return child_actions

    def is_terminal_state(self, state, player):
        check_worthy = [cell for cell in self.board.cells.values() if cell.pos[player - 1] == self.size - 1]
        for c in check_worthy:
            if c.owned == player and c.connected:
                return True
        return False

    def get_state_connected(self):
        return [(x.pos, x.connected) for x in self.board.cells.values()]


if __name__ == "__main__":
    size = 4

    game = HexGame(size)

    player = 1

    visualizer = HexMapVisualizer(game.board.cells.values(), True, size, game_type="hex")
    g = True
    while g:
        curr_state = game.get_state()
        visualizer.draw(curr_state, 0.0000001)

        actions = game.generate_possible_actions(curr_state)

        c = choice(actions)
        game.place_peg(c, player)

        if game.is_terminal_state(curr_state, player):
            print("Player {} won!".format(player))
            visualizer.draw(game.get_state(), 0.0000001)
            player = 1
            game = HexGame(size)
            visualizer = HexMapVisualizer(game.board.cells.values(), True, size, game_type="hex")

        player = 2 if player == 1 else 1

    print("Player {} won!".format(player))
    visualizer.draw(game.get_state(), 1000)

