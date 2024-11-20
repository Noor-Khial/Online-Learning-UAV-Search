# utils.py
import math
import random

def move_in_grid(index, direction, grid_size=6):
    """Moves the agent in the grid based on the specified direction.

    Args:
        index (int): The current index of the agent in a flattened grid.
        direction (int): The direction to move (0-7).
        grid_size (int): The size of the grid (default is 6).

    Returns:
        int: The new index after the move, or the same index if out of bounds.
    """
    # Define the direction mappings
    direction_mapping = {
        0: (-1, 0),  # Up
        1: (-1, 1),  # Up-right
        2: (0, 1),   # Right
        3: (1, 1),   # Down-right
        4: (1, 0),   # Down
        5: (1, -1),  # Down-left
        6: (0, -1),  # Left
        7: (-1, -1)  # Up-left
    }

    # Calculate current row and column
    row = index // grid_size
    col = index % grid_size
    move = direction_mapping.get(direction)

    if move is None:
        raise ValueError("Direction must be a number between 0 and 7.")

    # Calculate new position
    new_row = row + move[0]
    new_col = col + move[1]

    # Check bounds and return new index or stay in place
    if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
        new_index = new_row * grid_size + new_col
        return new_index
    else:
        return index  # Stay in the same position if out of bounds

def create_grid(biases, grid_size):
    """Creates a grid based on the given biases.

    Args:
        biases (list): A list of biases for the grid cells.
        grid_size (int): Size of the grid.

    Returns:
        list: A 2D grid representation of the biases.
    """
    grid = []
    g = 0
    for _ in range(grid_size):
        g_i = []
        for _ in range(grid_size):
            g_i.append(g)
            g += 1
        grid.append(g_i)
    return grid

def draw(weights):
    """Picks an index from the given list of weights proportionally.

    Args:
        weights (list): A list of weights.

    Returns:
        int: An index selected based on the weights.
    """
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1

def distr(weights, gamma=0.0):
    """Normalizes a list of weights to a probability distribution.

    Args:
        weights (list): A list of weights.
        gamma (float): An egalitarianism factor.

    Returns:
        tuple: A tuple representing the normalized probability distribution.
    """
    theSum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)
