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
    
    def create_neighborhood(self, cells):
        """
            Creates a neighborhood for each cell on the map. The neighborhood 
            pertains to the nature of a hexagonal grid.
        """
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
        return cells
    
    def init_map(self):
        """
            Initializes the map based on its characteristics:
                - Size (number of cells)
                - Type (Diamond or Triangle)
                - Occupied cells

            Creates neighborhood for each cell in map.
        """
        cells = []
        for i in range(self.size):
            for j in range(self.size if self.type == "diamond" else i+1):
                cells.append(Cell((i, j), 0))
                
        return self.create_neighborhood(cells)
        

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
        pos = tuple(row, column), a boolean "pegged" for knowing if the cell is
        pegged or empty, and a list of neighbors. 
    """
    def __init__(self, pos, pegged, neighbors=[]):
        self.pos = pos
        self.pegged = pegged
        self.neighbors = neighbors
    
    def add_neighbors(self, neighbors):
        self.neighbors = neighbors
    
    def set_peg(self, b):
        self.pegged = b