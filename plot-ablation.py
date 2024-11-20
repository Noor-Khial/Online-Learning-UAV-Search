from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

plt.rcParams["figure.dpi"] = 500
plt.style.use("ggplot")

# Function to extract a column from a matrix
def column(A, j):
    return [A[i][j] for i in range(len(A))]

# Function to transpose a matrix
def transpose(A):
    return [column(A, j) for j in range(len(A[0]))]

# Function to plot weak regret and regret bound
def regretWeightsGraph(filenames, title):
    # Initialize the plot
    plt.figure()

    # Iterate over each filename for weak regret
    for i, filename in enumerate(filenames):
        with open(filename, 'r') as infile:
            lines = infile.readlines()

        lines = [[eval(x.split(": ")[1]) for x in line.split('\t')] for line in lines]
        data = transpose(lines)

        regret = np.array(data[0])[0:175000]
        xs = np.array(list(range(len(data[0]))))[0:175000]
        plt.xlabel('Time slot ($\\times 10^{5}$)')
        plt.ylabel('$R_{T} (\\times 10^{4})$')
        colors = ['#ff0000', '#9467bd', '#e377c2','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#8c564b', '#7f7f7f', '#bcbd22', '#17becf',
          '#aec7e8', '#ffbb78', '#98df8a', '#c5b0d5']

        labels = [
          r'$|\mathcal{M}| = 1$',
          r'$|\mathcal{M}| = 3$', 
          r'$|\mathcal{M}| = 5$',
          r'$|\mathcal{M}| = 7$',
          r'$|\mathcal{M}| = 9$',
          ]
        plt.plot(xs/100000, regret/10000, label=f"{labels[i]}", color = colors[i], linewidth=2)
    plt.legend()
    plt.show()

# Example usage
filenames = [
             'exp3/alg/results/exp4|E|=1.txt',
             'exp3/alg/results/exp4|E|=3.txt',
             'exp3/alg/results/exp4|E|=5.txt',
             'exp3/alg/results/exp4|E|=7.txt',
             'exp3/alg/results/exp4|E|=9.txt',
             ] 
regretWeightsGraph(filenames, " ")

filenames = [
            'exp3/alg/results/final/c-exp3(|C|=9).txt',
             'exp3/alg/results/final/c-exp3(|C|=16).txt',
             'exp3/alg/results/final/c-exp3(|C|=36).txt', 
             'exp3/alg/results/final/exp3.txt'] 

labels = [
          r'C-Exp3, $|\mathcal{K}| = 9, |\mathcal{C}| = 9$',
          r'C-Exp3, $|\mathcal{K}| = 9, |\mathcal{C}| = 16$', 
          r'C-Exp3, $|\mathcal{K}| = 9, |\mathcal{C}| = 36$',
          r'Exp3, $|\mathcal{K}| = 36$ - Practically Infeasible']