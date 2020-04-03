from hexmap import DiamondHexMap
from random import choice
from visualization import HexMapVisualizer

class HexGame:
    def __init__(self, size):
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


if __name__ == "__main__":
    size = 4

    game = HexGame(size)

    player = 1

    visualizer = HexMapVisualizer(game.board.cells.values(), True, size)
    g = True
    while g:
        curr_state = game.get_state()
        visualizer.draw(curr_state, 0.5)
        #input(curr_state)
        actions = game.generate_possible_actions(curr_state)
        #input(actions)
        #input(c)
        if actions == []:
            g == False
            # visualizer.draw_win() 
            #   the winning path blinks a couple of times 
            #   to clearly show who won
            visualizer.draw(curr_state, 0.5)
        else:
            c = choice(actions)
            game.place_peg(c, player)
            player = 2 if player == 1 else 1

