# visualize_tour.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import load_model

MODEL_PATH = "outputs/tsp_20/modelAlpha_20250731T180107/epoch-0.pt"  # <- change if needed

# 1) Load model and set up device/decoding
model, _ = load_model(MODEL_PATH)
model.eval()
model.set_decode_type("greedy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2) Make one random TSP instance (coords: [graph_size, 2])
dataset = model.problem.make_dataset(num_samples=1)
sample = dataset[0]                      # torch.Tensor [N, 2]
sample = sample.to(device)               # put on same device as model
coords = sample.cpu().numpy()            # for plotting

# 3) Run the model to get a tour
with torch.no_grad():
    cost, ll, pi = model(sample.unsqueeze(0), return_pi=True)  # add batch dim

# 4) Extract the tour for the single instance
tour = pi[0].cpu().numpy()               # shape [N], node indices
tour = np.append(tour, tour[0])          # close the loop

# 5) Plot
plt.figure(figsize=(6, 6))
plt.plot(coords[tour, 0], coords[tour, 1], 'o-', markersize=8)
plt.title(f"TSP Tour (cost ~ {cost.item():.3f})")
plt.axis('equal')
plt.grid(True)
plt.show()
