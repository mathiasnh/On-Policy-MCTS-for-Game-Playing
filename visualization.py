import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt
from hexmap import DiamondHexMap, TriangleHexMap

class HexMapVisualizer:
    def __init__(self, cells, is_diamond_map, size):
        self.cells = cells
        self.is_diamond_map = is_diamond_map
        self.size = size
        self.board = self.create_graph()
        self.color_map = {0: "white", 1: "red", 2: "blue"}
        self.edge_colors, self.weights = self.set_border()

    def set_border(self):
        red_edge_indices, blue_edge_indices = self.generate_indices()
        colors = []
        weights = []
        for x, y, label in self.board.edges.data("label"):
            if label in red_edge_indices:
                colors.append("red")
                weights.append(3)
            elif label in blue_edge_indices:
                colors.append("blue")
                weights.append(3)
            else:
                colors.append("black")
                weights.append(1)

        return colors, weights

    def generate_indices(self):
        default_size = 3
        default_end = 5
        red_default = [0,2,14,15]
        blue_default = [1,8,6,13]

        if self.size == default_size:
            return red_default, blue_default
        
        red = []
        blue = []

        length = self.size - 1

        offset_0 = (self.size - default_size) * 3
        offset_1 = (self.size - default_size - 1) * 3

        for i in range(length):
            if i < 2:
                red.append(red_default[:3][i])
            else:
                red.append(red[-1] + 3)

            if i == 0:
                blue.append(1)
            else:
                blue.append(blue[-1] + 10 + offset_1)
        
        for i in range(length):
            red.append((default_end + offset_0) * self.size - i)

            if i == 0:
                blue.append(6 + offset_0)
            else:
                blue.append(blue[-1] + 10 + offset_1)
        

        return red, blue
        


    def create_graph(self):
        """
            Creates a graph using Networkx, adds nodes from a HexMap 
            and edges between neighboring nodes
        """
        i = 0
        g = nx.Graph()
        nodes = set()
        for cell in self.cells:
            g.add_node((cell.pos[0], cell.pos[1]), pos=self.pos_in_graph(cell.pos))
            nodes.add((cell.pos[0], cell.pos[1]))
            for neighbor in cell.neighbors:
                if (neighbor.pos[0], neighbor.pos[1]) not in nodes:
                    g.add_edge((cell.pos[0], cell.pos[1]), (neighbor.pos[0], neighbor.pos[1]), label=i)
                    i+=1    
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
        plt.pause(delay)
        plt.clf()
        nx.draw(
            self.board,
            nx.get_node_attributes(self.board, 'pos'),
            node_color=self.get_node_colors(state),
            edgecolors="black",
            edge_color=self.edge_colors,
            width=self.weights
        )
        """
        edge_labels=dict([((u,v,),d['label'])
             for u,v,d in self.board.edges(data=True)])
        nx.draw_networkx_edge_labels(
            self.board,
            nx.get_node_attributes(self.board, 'pos'),
            edge_labels=edge_labels)
        """
        plt.axis('equal')
        plt.draw()
