# visualize_tour.py

import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import load_model

# Set model path (change if needed)
MODEL_PATH = "outputs/tsp_20/modelAlpha_20250731T180107/epoch-0.pt"

# Load model
model, _ = load_model(MODEL_PATH)
model.eval()
model.set_decode_type("greedy")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create 1 sample instance
dataset = model.problem.make_dataset(num_samples=1)
sample = dataset[0].to(device)  # shape: [graph_size, 2]

# Save coordinates for plotting
coords = sample.cpu().numpy()

# Get tour
with torch.no_grad():
    cost, log_likelihood, pi = model(sample.unsqueeze(0), return_pi=True)  # Add batch dim

tour = pi[0].cpu().numpy()  # shape: [graph_size]
tour = np.append(tour, tour[0])  # close the loop

# Plot the tour and save
plt.figure(figsize=(6, 6))
plt.plot(coords[tour, 0], coords[tour, 1], 'o-', markersize=8)
plt.title(f"TSP Tour (cost ≈ {cost.item():.3f})")
plt.axis("equal")
plt.grid(True)
plt.savefig("tour_plot.png")
print("Tour plot saved to tour_plot.png")
