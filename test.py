import pickle
import matplotlib.pyplot as plt
import numpy as np

cost_path = 'results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl'
with open(cost_path, 'rb') as f:
    costs = pickle.load(f)

dataset_path = 'data/tsp/tsp_unif50_test_seed1234.pkl'
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

costs = np.array(costs)
print("Mean: ", np.mean(costs))
print("Median: ", np.median(costs))
print("Max: ", np.max(costs))
print("Min: ", np.min(costs))
print("Std: ", np.std(costs))

argsort = np.argsort(-costs) # Longest paths at the front
dataset = np.array(dataset)
dataset_sorted = dataset[argsort]




# Useless code that shows stuff
batch = dataset_sorted[:6]


# PLOT IT
def subplot_embedding(subplot, graph):
    subplot.scatter(graph[:,0], graph[:,1], color='black')
    #subplot.set_xticks([])
    #subplot.set_yticks([])

fig, axs = plt.subplots(2, 3, figsize=(12, 4 * 2))
plt.subplots_adjust(hspace=0.1)

for i in range(2):
    for j in range(3):
        subplot_embedding(axs[i, j], batch[i*3+j])

plt.show()