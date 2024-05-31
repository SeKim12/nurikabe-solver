from mcts import MCTS
from .nurikabe_grid_alpha import NurikabeGridAlpha
import numpy as np
from tensorflow.keras.models import load_model


class AlphaNurikabe(MCTS):
    """ Monte Carlo Tree Search for Nurikabe game.

    Terminal (leaf) nodes are actions that lead to an incorrect
    nurikabe grid (with 2 same value on one line for example).
    The value of a leaf node will be the number of cells filled.

    The policy to choose random actions is a convolutional network
    trained on 1 million nurikabe games.

    Note : unlike AlphaGo, I keep uct without probabilities to
    balance between exploration and exploitation. """

    def __init__(self, nurikabe_grid, solution, max_iterations=10000,
                 pathnet='./policy_network',
                 model=None, exploration_weight=1, max_depth_tree=10):

        """ Nurikabe Grid : either NurikabeGridAlpha with model initialised,
        or numpy array. If it is a numpy array, pathnet or directly model must
        be provided. """

        if isinstance(nurikabe_grid, np.ndarray):
            if model is None:
                model = load_model(pathnet)
            nurikabe_grid = NurikabeGridAlpha(nurikabe_grid, solution, model)
        super().__init__(nurikabe_grid, solution, exploration_weight, max_depth_tree,
                         max_iterations)
        self.probas = {}

    def _action_selection(self, nurikabe_grid):
        # All children of node should already be expanded:
        assert all(child in self.children
                   for child in self.children[nurikabe_grid])

        def to_maximise(child):
            """ Function to minimize described in original alphaGo paper. """
            return self.Q[child] / self.N[child] + child.proba_taken / \
                (1 + self.N[child])

        return max(self.children[nurikabe_grid], key=to_maximise)
