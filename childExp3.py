import math
import numpy as np
import sys
import os
from probability import distr, draw
from utils import move_in_grid
import PRGM


class Exp3Algorithm:
    def __init__(self, numActions, gamma, rewardMin=0, rewardMax=1):
        """Initializes the Exp3 algorithm parameters.

        Args:
            numActions (int): The number of possible actions.
            gamma (float): The exploration parameter.
            rewardMin (float): The minimum possible reward.
            rewardMax (float): The maximum possible reward.
        """
        self.numActions = numActions
        self.gamma = gamma
        self.weights = [1.0] * numActions
        self.cumulativeReward = 0
        self.bestActionCumulativeReward = 0
        self.rewardMin = rewardMin
        self.rewardMax = rewardMax
        self.biases = [0.0] * numActions
        self.grid = self.create_grid(self.biases)
        self.weakRegret = 0 
        self.regretBound = 0
        self.index = 0 
        self.best_action = 25 
        self.target_distribution = 0

    def create_grid(self, biases):
        """Creates a grid based on the given biases.

        Args:
            biases (list): A list of biases for the grid cells.

        Returns:
            list: A 2D grid representation of the biases.
        """
        grid = []
        g = 0
        size = int(math.sqrt(len(biases)))
        for i in range(size):
            g_i = []
            for j in range(size):
                g_i.append(g)
                g += 1
            grid.append(g_i)
        return grid

    def exp3(self, t, agent_location):
        """Performs one step of the Exp3 algorithm.

        Args:
            t (int): The time step or iteration number.
            agent_location (int): The current location of the agent.

        Returns:
            tuple: The selected action, the received reward, and the estimated reward.
        """
        # Get the probability distribution based on weights
        probabilityDistribution = distr(self.weights, self.gamma)
        choice = draw(probabilityDistribution)

        # Get the reward based on the new location
        theReward = self.get_reward(move_in_grid(agent_location, choice))
        scaledReward = (theReward - self.rewardMin) / (self.rewardMax - self.rewardMin)
        
        # Calculate the estimated reward
        estimatedReward = scaledReward / probabilityDistribution[choice]

        # Update the weights based on the estimated reward
        self.weights[choice] *= math.exp(estimatedReward * self.gamma / self.numActions)   
        return choice, theReward, estimatedReward

    def get_reward(self, agent_location):
        """Retrieves the reward for a given agent location.

        Args:
            agent_location (int): The current location of the agent.

        Returns:
            float: The reward associated with the current location.
        """
        return self.biases[agent_location]

    def update_biases(self, t):
        """Updates the biases based on the current time step.

        Args:
            t (int): The current time step.
        """
        self.biases, index = PRGM.updateL(t % 40, self.target_distribution)
    
    def calculate_regret(self, t):
        """Calculates the weak regret and regret bound.

        Args:
            t (int): The current time step.

        Returns:
            tuple: The weak regret and regret bound.
        """
        self.weakRegret = self.bestActionCumulativeReward - self.cumulativeReward
        self.regretBound = 2 * math.sqrt(t * self.numActions * math.log(self.numActions))
        return self.weakRegret, self.regretBound

    def get_regret(self):
        """Returns the current weak regret and regret bound.

        Returns:
            tuple: The weak regret and regret bound.
        """
        return self.weakRegret, self.regretBound
