"""
This file describes a point reference group mobility pattern used to simulate leader movement within a grid environment. 
The leader's movement is sampled from different normal distributions, each representing a distinct mobility pattern. 
The code defines the grid layout and calculates the next position of the leader based on adjacent grid points. 
It also updates the probability biases for possible leader actions, which can be used for decision-making in the context 
of UAV path planning or similar applications.
"""
#Note: we have used a generated leader trajectory and stored them in the PRGM.txt file. 
#This trajectory is used for all the simulations, to have a fair comparison. 
import math
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 500

def sample_from_distribution(size, mean, std_dev):
    """Samples from a normal distribution."""
    return np.random.normal(mean, std_dev, size).astype(int)

def getAdjacent(arr, i, j):
    n = len(arr) - 1
    m = len(arr[0]) - 1
    v = []

    for dx in range(-1 if (i > 0) else 0, 2 if (i < n) else 1):
        for dy in range(-1 if (j > 0) else 0, 2 if (j < m) else 1):
            if (dx != 0 or (dx + i) != n or dy != 0 or (dy + j) != m):
                v.append(arr[i + dx][j + dy])
    return v

def leaderMovement(time_step, pattern):
    location = 0 #start from the tope left of the grid
    
    # Sampled indices from distributions
    distributions = {
        0: sample_from_distribution(50, 25, 5),  
        1: sample_from_distribution(50, 10, 3), 
        2: sample_from_distribution(50, 20, 5) 
    }

    # Ensure that the sampled indices are within valid range (0-35)
    distributions = {k: np.clip(v, 0, 35) for k, v in distributions.items()}

    if pattern == 1:
        location = distributions[1][time_step]
    elif pattern == 2:
        location = distributions[2][time_step]
    elif pattern == 0:
        location = distributions[0][time_step]

    return location

def getNextL(i, h):
    grid = Grid(biases=[0.0] * 36)
    random_l = leaderMovement(i, h)
    for i in range(len(grid)):
        for m in range(len(grid)):
            if grid[i][m] == random_l:
                index = [i, m]
    return index, random_l

def updateL(i, h):
    numActions = 36
    grid = Grid(biases=[0.0] * 36)
    index, nextL = getNextL(i, h)
    biases = [0.0] * numActions
    adjacents = getAdjacent(grid, index[0], index[1])
    biases[nextL] = 1
    for i in range(len(adjacents)):
        biases[adjacents[i]] = 1
    return biases, index

def Grid(biases):
    grid = []
    g = 0
    for i in range(int(math.sqrt(len(biases)))):
        g_i = []
        for j in range(int(math.sqrt(len(biases)))):
            g_i.append(g)
            g = g + 1
        grid.append(g_i)
    return grid

#numActions = 6 * 6
#benchmark = [0.0] * numActions
#numRounds = 50000
#biases = [0.0] * numActions
#grid = Grid(biases)
#
#for i in range(numRounds):
#    biases, index = updateL(i % 40, 0)
#    for m in range(numActions):
#        benchmark[m] += biases[m]
#
#print(max(benchmark))
#print(benchmark.index(max(benchmark)))
#
## Dataset for plotting
#y = np.array(benchmark) / 35000
#x = np.array(range(0, 36))
#
#X_Y_Spline = make_interp_spline(x, y)
#
## Returns evenly spaced numbers over a specified interval.
#X_ = np.linspace(x.min(), x.max(), 500)
#Y_ = X_Y_Spline(X_)
#
## Plotting the Graph
#plt.plot(X_, Y_, color='navy')
#plt.ylabel('Probability Distribution')
#plt.xlabel('Action k')
#plt.show()
