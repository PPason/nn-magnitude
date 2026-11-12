import numpy as np
from matplotlib import pyplot as plt

# 0 = mean abs
# 1 = norm
metric_params = 1
metric_params_names = ["Mean abs", "Relative norm change"]

# 0 = max performance loss
# 1 = duration
metric_magnitude = 0
metric_magnitude_names = ["Maximum performance loss", "Duration"]

results_params_net = np.load("results_params_net.npy")
results_magnitudes = np.load("results_magnitude.npy")

magnitudes = np.mean(results_magnitudes, axis=0)

chunks = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

n_reps = 50

fig, ax = plt.subplots()
fig.tight_layout()

ax.scatter(magnitudes[:, metric_magnitude], np.mean(results_params_net[:, :, metric_params], axis=0))
ax.set_ylabel(f"{metric_params_names[metric_params]}")
ax.set_xlabel(f"{metric_magnitude_names[metric_magnitude]}")
ax.set_title("Mean results")
ax.grid()
ax.spines["top"].set_visible(0)
ax.spines["right"].set_visible(0)
ax.spines["left"].set_visible(0)
ax.spines["bottom"].set_visible(0)

fig1, ax1 = plt.subplots()
fig1.tight_layout()

for i in range(results_params_net.shape[1]):
    ax1.scatter([i] * n_reps, results_params_net[:, i, metric_params])

magnitudes = list(magnitudes[:, metric_magnitude])

ax1.set_xticklabels(np.round(magnitudes, 3))

ax1.set_xlabel(f"{metric_magnitude_names[metric_magnitude]}")
ax1.set_ylabel(f"{metric_params_names[metric_params]}")
ax1.set_title("Results")
ax1.grid()
ax1.spines["top"].set_visible(0)
ax1.spines["right"].set_visible(0)
ax1.spines["left"].set_visible(0)
ax1.spines["bottom"].set_visible(0)

fig1.savefig("experiment2_whole_network.png")

plt.show()