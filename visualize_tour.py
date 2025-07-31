import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import load_model

# Load model
model, _ = load_model("outputs/tsp_20/modelAlpha_20250731T180107/epoch-0.pt")
model.eval()
model.set_decode_type("greedy")

# Generate 1 random instance
dataset = model.problem.make_dataset(num_samples=1)
sample = dataset[0]  # Tensor of shape [graph_size, 2]
coords = sample.numpy()

# Get tour
with torch.no_grad():
    tour, _ = model(sample.unsqueeze(0), return_pi=True)  # Add batch dimension

tour = tour.squeeze().numpy()

# Close the tour loop
tour = np.append(tour, tour[0])

# Plot
plt.figure(figsize=(6, 6))
plt.plot(coords[tour, 0], coords[tour, 1], 'o-', markersize=8)
plt.title("TSP Tour")
plt.grid(True)
plt.show()
