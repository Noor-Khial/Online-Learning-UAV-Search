import numpy as np
import math
from exp3 import Exp3Algorithm
import PRGM
from probability import distr, draw
from utils import create_grid, move_in_grid

class ContextExp3:
    def __init__(self, numContexts, numActions, numRounds, gamma, grid_size=6):
        self.numContexts = numContexts
        self.numActions = numActions
        self.numRounds = numRounds
        self.grid_size = grid_size
        self.contexts = [Exp3Algorithm(numActions, gamma) for _ in range(self.numContexts)]
        self.totalWeakRegret = 0
        self.totalRegretBound = 0
        self.biases = [0.0] * numActions
        self.grid = create_grid(self.biases, grid_size)
        self.agent_location = 0
        self.best_action = 25
        self.target_distribution = 0

    def get_context(self, cell_number, n):
        """Get the context number based on the cell number and the number of subgrids."""
        num_subgrids_per_row = self.grid_size // n

        # Convert cell number to row and column (1-indexed)
        row = (cell_number - 1) // self.grid_size
        col = (cell_number - 1) % self.grid_size

        # Convert to subgrid row and column
        subgrid_row = row // n
        subgrid_col = col // n

        # Calculate subgrid number (1-indexed)
        subgrid_number = subgrid_row * num_subgrids_per_row + subgrid_col + 1
        return subgrid_number

    def run(self, t):
        """Run one iteration of the Context Exp3 algorithm."""
        with open('weights/exp3-test.txt', 'a') as f:
            context_regrets = []
            context_bounds = []

            # Update biases and get context number
            biases, _ = PRGM.updateL(t % 40, self.target_distribution)
            context_number = self.get_context(self.agent_location, n=1)

            for i, context in enumerate(self.contexts):
                weak_regret, regret_bound = context.get_regret()
                context.best_action = self.best_action
                context.target_distribution = self.target_distribution

                if context_number == i:
                    context.update_biases(t)
                    action, reward, est = context.exp3(t, self.agent_location)
                    context.cumulativeReward += reward
                    context.bestActionCumulativeReward += biases[self.best_action]
                    weak_regret, regret_bound = context.calculate_regret(t)
                    self.agent_location = move_in_grid(self.agent_location, action)
                    self.weights = distr(context.weights)
                    self.est = reward

                context_regrets.append(weak_regret)
                context_bounds.append(regret_bound)

            self.totalWeakRegret = np.sum(context_regrets)
            self.totalRegretBound = np.sum(context_bounds)

            # Log the results
            m = f"regret: {self.totalWeakRegret:.3f}\tweights: ({', '.join([f'{weight:.3f}' for weight in self.weights])})"
            f.write(m + '\n')

    @staticmethod
    def simple_test(numRounds=40000, gamma=None, target_updates=None):
        """Run a simple test of the ContextExp3 algorithm with customizable parameters."""
        numContexts = 36  # Example number of contexts
        numActions = 8  # Number of actions

        # Default gamma if not provided
        if gamma is None:
            gamma = math.sqrt(math.log(numRounds) / numRounds * numActions)
        context_exp3 = ContextExp3(numContexts, numActions, numRounds, gamma)
        # Set target updates if provided
        if target_updates is None:
            target_updates = {
                40000: (1, 10),
                200000: (2, 25),
                1500000: (0, 19)
            }

        with open('weights/exp3-test.txt', 'w') as f:
            for t in range(numRounds):
                context_exp3.run(t)

                # Update target distribution and best action based on the round number
                if t in target_updates:
                    context_exp3.target_distribution, context_exp3.best_action = target_updates[t]

if __name__ == "__main__":
    ContextExp3.simple_test()
