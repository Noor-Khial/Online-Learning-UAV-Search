import numpy as np
import math
import PRGM 
from cexp3 import ContextExp3
from utils import move_in_grid

class Exp4:
    def __init__(self, num_actions, num_experts, eta, num_contexts, num_rounds):
        self.num_actions = num_actions
        self.num_experts = num_experts
        self.eta = eta
        self.biases = [0.0] * num_actions
        self.weights = np.ones(num_experts) / num_experts
        self.agent_location = 0
        self.t = 0
        
        #logarithmic search 
        self.gamma = [math.sqrt(math.log(num_rounds * factor) / (num_rounds * factor) * num_actions) for factor in [1, 10, 0.1, 100, 0.01, 1000, 0.001, 10000, 0.0001]]
        self.gamma= self.gamma[:2]
        self.experts = [self.create_expert(num_contexts, num_actions, num_rounds, gamma_value) for gamma_value in self.gamma]
        self.totalWeakRegret = 0
        self.best_action = None  # Initialized later
        self.targets_distribution = 0

    def create_expert(self, num_contexts, num_actions, num_rounds, gamma):
        return ContextExp3(num_contexts, num_actions, num_rounds, gamma)

    def get_action_probabilities(self, expert_predictions):
        combined_predictions = np.dot(self.weights, expert_predictions)
        action_probabilities = combined_predictions / np.sum(combined_predictions)
        return action_probabilities
    
    def update(self, t):
        index_of_highest = np.argmax(self.weights)
        self.agent_location = self.experts[index_of_highest].agent_location
        reward = self.get_reward(self.agent_location, t)
        estimated_rewards = np.array([e.est for e in self.experts])
        exponentiated_weights = np.exp(self.eta * estimated_rewards) * self.weights
        self.weights = exponentiated_weights / np.sum(exponentiated_weights)

        self.totalWeakRegret += (self.biases[self.best_action] - reward)
        self.log_regret_and_weights()
        
    def log_regret_and_weights(self):
        with open('weights/exp4-test.txt', 'a') as f:
            f.write(f"regret: {self.totalWeakRegret:.3f}\tweights: ({', '.join(['%.3f' % weight for weight in self.weights])})\n")

    def update_biases(self, t):
        self.biases, index = PRGM.updateL(t % 50, self.targets_distribution)

    def select_action(self, expert_predictions):
        action_probabilities = self.get_action_probabilities(expert_predictions)
        action = np.random.choice(range(self.num_actions), p=action_probabilities)
        return action
    
    def get_reward(self, agent_location, t):
        self.t = t
        self.update_biases(t)
        return self.biases[agent_location]
    
    def get_experts_prediction(self):
        for expert in self.experts:
            expert.run(self.t)
        expert_predictions = np.array([expert.weights for expert in self.experts])
        expert_predictions /= expert_predictions.sum(axis=1, keepdims=True)
        return expert_predictions


# Parameters
num_actions = 8
num_experts = 2
eta = 0.0001
num_contexts = 36  # Number of contexts
num_rounds = 200000  # Number of rounds

# Initialize the Exp4 algorithm
exp4 = Exp4(num_actions, num_experts, eta, num_contexts, num_rounds)

# Main loop for iterations
for iteration in range(num_rounds):

    # Dynamic updates for best actions and target distribution
    if iteration in {0, 50000, 100000, 150000, 200000}:
        exp4.best_action = {0: 25, 50000: 10, 100000: 19, 150000: 25, 200000: 10}[iteration]
        exp4.targets_distribution = {0: 2, 50000: 1, 100000: 2, 150000: 0, 200000: 1}[iteration]
        for e in exp4.experts:
            e.best_action = exp4.best_action
            e.target_distribution = exp4.targets_distribution

    expert_predictions = exp4.get_experts_prediction()    
    action = exp4.select_action(expert_predictions)
    exp4.update(iteration)
