import numpy as np

class NurikabeGrid:
    actions = {-1, 0}

    def __init__(self, world, soln):
        self.world = world
        self.soln = soln
    
    def update(self, r, c, a):
        assert a in self.actions
        self.world[r, c] = a
        return self.get_reward()
    
    def is_solved(self):
        return self.soln == self.world

    def get_reward(self):
        # TODO: come up with better reward
        # currently just "distance" w/ reference solution
        return -np.linalg.norm(self.world - self.soln, ord='fro')
    
    def get_current_state(self):
        return self.world
    
    def get_num_rows(self):
        return len(self.world)

    def get_num_cols(self):
        return len(self.world[0])