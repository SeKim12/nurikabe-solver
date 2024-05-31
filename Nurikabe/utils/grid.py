import numpy as np
from .utils import SIZE


class Grid:
    """ Implements a minimalist version of a Grid for
    nurikabe. Only a function to fill empty cells and another
    to check validity of the grid are provided. """

    # we can always ensure that grid is np.ndarray and not None
    def __init__(self, grid=None, solution=None):
        self.solution = solution
        if isinstance(grid, str):
            grid = np.array([int(i) if i != '.' else 0
                             for i in grid])
            grid = grid.reshape((SIZE ** 2, SIZE ** 2))
        self.grid = grid if grid is not None \
            else np.zeros((SIZE ** 2, SIZE ** 2))

    def fill_cell(self, i, j, value):
        # if self.grid[i, j] == -2:
        if self.grid[i, j] < 1:
            self.grid[i, j] = value
        else:
            raise ValueError("The cell you are trying to fill is "
                             "not emply")

    def is_correct(self, solution):
        # check 
        for i in range(9):
            for j in range(9):
                if self.grid[i, j] != -2 and \
                    self.grid[i, j] != solution[i, j]:
                    return False
        return True
    
        # for value in range(1, SIZE ** 2 + 1):
        #     for i in range(SIZE ** 2):
        #         if (self.grid[i, :] == value).sum() > 1:
        #             return False
        #         elif (self.grid[:, i] == value).sum() > 1:
        #             return False
        #     for i in range(SIZE ** 2):
        #         for j in range(SIZE ** 2):
        #             if (self._values_in_box(SIZE * i, SIZE * j) ==
        #                     value).sum() > 1:
        #                 return False
        # return True

    def is_complete(self):
        """ Check only completeness not correctness. """
        return not np.any(self.grid == -2)

    def copy(self):
        return Grid(self.grid.copy())

    def _values_in_box(self, i, j):
        """ All values in the square SIZE * SIZE. """
        idx_line, idx_col = SIZE * (i // SIZE), SIZE * (j // SIZE)
        return self.grid[idx_line: idx_line + SIZE,
                         idx_col: idx_col + SIZE]


class SmartGrid(Grid):
    """ More complete class to represent a Nurikabe Grid. The main
    difference is the calculation and storage of possibilities
    for each cell, and the possibility to go back. """

    def __init__(self, grid=None, solution=None):
        super().__init__(grid, solution)
        self.possibilities = {}

    @classmethod
    def from_grid(cls, grid, solution):
        if isinstance(grid, str):
            grid = np.array([int(i) if i != '.' else 0
                             for i in grid])
            grid = grid.reshape((SIZE ** 2, SIZE ** 2))
        obj = cls(grid)
        obj.possibilities = obj._pos()
        # breakpoint()
        return obj

    def fill_cell(self, i, j, value):
        self.grid[i, j] = value
        del self.possibilities[(i, j)]  # raises error if not present
        # for index in self._related_indeces(i, j):
        #     pos_at_index = self.possibilities.get(index, [])
        #     if value in pos_at_index:
        #         pos_at_index.remove(value)

    def erase_cell(self, i, j):
        self.grid[i, j] = -2
        self.possibilities = self._pos()

    def index_with_min_pos(self):
        min_pos = min(self.possibilities.values(), key=len)
        return [k for k, v in self.possibilities.items() if v == min_pos]

    def copy(self):
        new_grid = SmartGrid(self.grid.copy())
        new_grid.possibilities = self.possibilities.copy()
        return new_grid

    def _related_indeces(self, i, j):
        """ All indeces that will be impacted by changing (i, j). """
        indices = self._indeces_in_box(i, j)
        for k in range(SIZE ** 2):
            if (i, k) not in indices:
                indices.append((i, k))
            if (k, j) not in indices:
                indices.append((k, j))
        return indices

    def _pos(self):
        """ Return a dictionary with keys being index and
        values being a list of different possibilities. """

        line, col = np.where(self.grid == -2)
        pos = {(line[i], col[i]): self._pos_at(line[i], col[i])
               for i in range(len(line))}
        return pos
    
    def check_2_by_2(self, row, col):
        rows, cols = self.grid.shape

        blocks = [
            (row - 1, col - 1), (row - 1, col), 
            (row, col - 1), (row, col)
        ]
        
        for r, c in blocks:
            if 0 <= r < rows - 1 and 0 <= c < cols - 1:
                block = self.grid[r:r + 2, c:c + 2]
                # Temporarily set (row, col) to -1 for the check
                temp_block = block.copy()
                if row in range(r, r+2) and col in range(c, c+2):
                    temp_block[row - r, col - c] = -1
                if np.all(temp_block == -1):
                    return True

        return False
    

    def count_num_islands(self, i, j):
        visited = set()
        if (i, j) in visited:
            return
        cell_number = None
        
        def dfs(x, y):
            if (x, y) in visited:
                return
            visited.add((x, y))
            if (cell_number is None) and self.grid[x, y] >= 1:
                cell_number = self.grid[x, y]
            elif isinstance(cell_number, int) and self.grid[x, y] >= 1:
                return -1
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0]) \
                    and self.grid[nx, ny] == 0:
                    dfs(nx, ny)
        
        dfs(i, j)
        return len(visited)
    

    def _pos_at(self, i, j):
        """ Possibilities at index (i, j). """

        all_possibilities = [-1, 0]

        # check for 2x2
        if self.check_2_by_2(i, j):
            return [0]
        
        # self.grid[i, j] = 0
        # if self.count_num_islands(i, j) == -1:
        #     return [0]
        # self.grid[i, j] = -2
        
        return [-1, 0]

        # local_pos = list(range(1, SIZE ** 2 + 1))
        # to_delete = []
        # for pos in local_pos:
        #     if pos in self._values_in_box(i, j):
        #         to_delete.append(pos)
        #     elif pos in self.grid[i, :]:
        #         to_delete.append(pos)
        #     elif pos in self.grid[:, j]:
        #         to_delete.append(pos)
        # return [pos for pos in local_pos if pos not in to_delete]

    def _indeces_in_box(self, i, j):
        idx_line, idx_col = SIZE * (i // SIZE), SIZE * (j // SIZE)
        indeces = []
        for i in range(SIZE):
            for j in range(SIZE):
                indeces.append((idx_line + i, idx_col + j))
        return indeces
