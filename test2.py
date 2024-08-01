import json
import matplotlib.pyplot as plt


# Utility function for reading a json file
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

folder = 'results/tsp/tsp_unif50_test_seed1234/'
field = 'Gap_Rel_Avg'
models = ['test_vanilla', 'test_hac', 'test_dcd_hac']
#models = ['test_vanilla_decay', 'test_dcd_decay', 'test_hac_decay']
models = ['ewc_01', 'ewc_001', 'ewc_0001'] + models

data = {}
for model in models:
    file_path = folder + model + '-epoch_data.json'
    data[model] = read_json_file(file_path)

epochs = data[models[0]].keys()
epochs = [int(epoch) for epoch in epochs]
epochs.sort()

for model in models:
    plt.plot(epochs, [float(data[model][str(epoch)][field]) for epoch in epochs], label=model)

plt.xlabel('Epoch')
plt.ylabel(field)
plt.legend()
plt.show()
