import random

class HexMap:
    def __init__(self, 
                type,
                offsets, 
                size,
                tot_cell_count,
                rand_start, 
                rand_start_num, 
                fixed_open_cells):
        self.cells = None
        self.type = type
        self.offsets = offsets
        self.size = size
        self.tot_cell_count = tot_cell_count
        self.rand_start = rand_start
        self.rand_start_num = rand_start_num
        self.fixed_open_cells = fixed_open_cells
    
    def create_neighborhood(self):
        """
            Creates a neighborhood for each cell on the map. The neighborhood 
            pertains to the nature of a hexagonal grid.
        """
        cells = list(self.cells.values())
        for cell in cells:
            r, c = cell.pos
            neighbors = []
            # self.offsets is dependent on the board type and affects the 
            # nature of the neighborhood
            for (a,b) in self.offsets: 
                if r+a >= 0 and c+b >= 0:
                    n = next((x for x in cells if x.pos == (r+a, c+b)), None)
                    if n != None:
                        neighbors.append(n)

            cell.add_neighbors(neighbors)
    
    def init_map(self):
        """
            Initializes the map based on its characteristics:
                - Size (number of cells)
                - Type (Diamond or Triangle)
                - Random or fixed empty/open cells at start
                - If fixed, which cells to be empty/open

            Creates neighborhood for each cell in map.
        """
        self.cells = {}
        for i in range(self.size):
            for j in range(self.size if self.type == "diamond" else i+1):
                #self.cells.append(Cell((i, j), 0))
                self.cells[(i,j)] = Cell((i, j), 0)
        self.create_neighborhood()
        

class DiamondHexMap(HexMap):
    """
        Diamond shape hexagonal grid with structure as described in 
        the PDF Board Games on Hexagonal Grids
    """
    def __init__(self, 
                size, 
                rand_start, 
                rand_start_num, 
                fixed_open_cells=[]):
        super().__init__("diamond",
                        [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)],
                        size,
                        size**2,
                        rand_start,
                        rand_start_num,
                        fixed_open_cells)

class TriangleHexMap(HexMap):
    """
        Triangle shape hexagonal grid with structure as described in 
        the PDF Board Games on Hexagonal Grids
    """
    def __init__(self,
                size,
                rand_start, 
                rand_start_num, 
                fixed_open_cells=[]):
        super().__init__("triangle", 
                        [(-1,-1), (-1,0), (0,-1), (0,1), (1,0), (1,1)],
                        size,
                        sum(x for x in range(size+1)),
                        rand_start,
                        rand_start_num,
                        fixed_open_cells)

class Cell:
    """
        An individual cell in the hexagonal grid. Has a positional attribute 
        pos = tuple(row, column), a value "owner" for knowing if the cell is
        empty or owned by player 1 (red) or 2 (blue), and a list of neighbors. 
    """
    def __init__(self, pos, pegged, neighbors=[]):
        self.pos = pos
        self.owned = pegged
        self.neighbors = neighbors
        self.role = None
        self.connected = False
    
    def add_neighbors(self, neighbors):
        self.neighbors = neighbors

    def set_peg(self, player):
        self.owned = player
        self.check_placement(player)
        self.propogate_connection(player)

    def check_placement(self, player):
        x, y = self.pos
        if player == 1 and x == 0:
            self.connected = True
        elif player == 2 and y == 0:
            self.connected = True
        elif [n for n in self.neighbors if n.connected and n.owned == player] != []:
            self.connected = True

    def propogate_connection(self, player):
        for neighbor in self.neighbors:
            if self.connected and neighbor.owned == player and not neighbor.connected:
                    """
                    print("{} connected to {}".format((self.pos, "red" if self.owned == 1 else "blue"),\
                        (neighbor.pos, "red" if neighbor.owned == 1 else "blue")))
                    """
                    neighbor.connected = True 
                    neighbor.propogate_connection(player)


            