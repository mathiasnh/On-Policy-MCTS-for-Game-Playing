import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt
from hexmap import DiamondHexMap, TriangleHexMap

class HexMapVisualizer:
    def __init__(self, cells, is_diamond_map, size, game_type):
        self.cells = cells
        self.is_diamond_map = is_diamond_map
        self.size = size
        self.board = self.create_graph(size-1, game_type)
        self.edges = self.board.edges()
        self.colors = [self.board[u][v]['color'] for u,v in self.edges]
        self.weights = [self.board[u][v]['weight'] for u,v in self.edges]
        self.color_map = {0: "white", 1: "red", 2: "blue"}

    def create_graph(self, size, game_type):
        """
            Creates a graph using Networkx, adds nodes from a HexMap 
            and edges between neighboring nodes
        """
        g = nx.Graph()
        red_pairs = set()
        blue_pairs = set()
        black_pairs = set()

        for cell in self.cells:
            g.add_node((cell.pos[0], cell.pos[1]), pos=self.pos_in_graph(cell.pos))

            for neighbor in cell.neighbors:
                if cell.pos[0] == 0 and neighbor.pos[0] == 0 or cell.pos[0] == size and neighbor.pos[0] == size:
                    red_pairs.add(((cell.pos[0], cell.pos[1]), (neighbor.pos[0], neighbor.pos[1])))
                elif cell.pos[1] == 0 and neighbor.pos[1] == 0 or cell.pos[1] == size and neighbor.pos[1] == size:
                    blue_pairs.add(((cell.pos[0], cell.pos[1]), (neighbor.pos[0], neighbor.pos[1])))
                else:
                    black_pairs.add(((cell.pos[0], cell.pos[1]), (neighbor.pos[0], neighbor.pos[1])))
            
        if game_type == "hex":
            g.add_edges_from(red_pairs, color="r", weight=4)
            g.add_edges_from(blue_pairs, color="b", weight=4)
            g.add_edges_from(black_pairs, color="black", weight=1)
        else:
            g.add_edges_from([((cell.pos[0], cell.pos[1]), (neighbor.pos[0], neighbor.pos[1])) \
                for cell in self.cells for neighbor in cell.neighbors])
            
        return g

    def pos_in_graph(self, pos):
        """
            Calculates the positions of a node in the graph based on its position
            on the board. Dependent on map type (Diamond or Triangle).
        """
        r, c = pos
        if self.is_diamond_map:
            return (c - r) / 2, -(c + r) * sqrt(3) / 2
        else:
            return c - r / 2, -r * sqrt(3) / 2 

    def get_node_colors(self, state):
        """
            Returns a list of colors to be matched with their respective nodes in 
            a graph. Green is node is to jump, red if node is to be jumped over, 
            black if node is pegged, and white if node is empty.
        """
        colors = []
        for cell in state:
            pos, owned = cell
            colors.append(self.color_map[owned])
        return colors

    def draw(self, state, delay, show=False):
        """
            Draw a given state with the current board-graph. The action parameter 
            contains references to the jumper and jumped (green and red respectively).
        """
        plt.clf()
        nx.draw(
            self.board,
            nx.get_node_attributes(self.board, 'pos'),
            node_color=self.get_node_colors(state),
            edgecolors="black",
            edge_color=self.colors, 
            width=self.weights
        )
        plt.axis('equal')
        plt.draw()
        if show:
            plt.show()
        else:
            plt.pause(delay)
