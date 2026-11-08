import numpy as np
from matplotlib import pyplot as plt

# 0 = mean abs
# 1 = norm
metric_params = 1
metric_params_names = ["mean abs", "relative norm change"]

# 0 = max performance loss
# 1 = duration
metric_magnitude = 0
metric_magnitude_names = ["max performance loss", "duration"]

results_params_layers = np.load("results_params.npy")
results_magnitudes = np.load("results_magnitude.npy")

magnitudes = np.mean(results_magnitudes, axis=0)

chunks = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

n_layers = 4

n_reps = 50

for n_layer in range(n_layers):
    fig, ax = plt.subplots()
    ax.scatter(magnitudes[:, metric_magnitude], np.mean(results_params_layers[:, n_layer, :, metric_params], axis=0))
    ax.set_ylabel(f"{metric_params_names[metric_params]}")
    ax.set_xlabel(f"Magnitude ({metric_magnitude_names[metric_magnitude]})")
    ax.set_title(f"Mean results | Layer {n_layer}")

    fig1, ax1 = plt.subplots()

    for i in range(results_params_layers.shape[2]):
        ax1.scatter([i] * n_reps, results_params_layers[:, n_layer, i, metric_params])

    magnitudes_list = list(magnitudes[:, metric_magnitude])

    ax1.set_xticklabels(np.round(magnitudes_list, 3))

    ax1.set_xlabel(f"Magnitude ({metric_magnitude_names[metric_magnitude]})")
    ax1.set_ylabel(f"{metric_params_names[metric_params]}")
    ax1.set_title(f"Results | Layer {n_layer}")

colors = ["red", "blue", "green", "orange"]

fig1, ax1 = plt.subplots()

offsets = (np.arange(n_layers) - (n_layers - 1) / 2) * 0.8 / n_layers

for n_ch, n_chunk in enumerate(chunks[1:]):
    for i in range(n_layers):
        ax1.scatter([n_ch + offsets[i]] * n_reps, results_params_layers[:, i, n_ch, metric_params ], color=colors[i], label=f"Layer {i}" if n_ch == 0 else None)

ax1.set_xticks(range(len(chunks)))
ax1.set_xticklabels(np.round(magnitudes[:, 0], 3))

ax1.set_xlabel("Magnitude")
ax1.set_ylabel(metric_params_names[metric_params])
ax1.set_title("Results")

ax1.legend()

plt.show()