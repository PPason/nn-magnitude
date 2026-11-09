import numpy as np
from matplotlib import pyplot as plt

# 0 = abs mean
# 1 = norm
metric = 1

metric_names = ["abs mean", "relative norm change"]

magnitude_max = 5
magnitude_step = 0.1

magnitudes = np.linspace(0, magnitude_max, int(round(magnitude_max / magnitude_step)) + 1)

n_reps = 50

n_layers = 4

results = np.load("results_layers.npy")

magnitudes_list = [x for x in magnitudes for y in range(n_reps)]

results_mean = np.mean(results, axis=0)
results_std = np.std(results, axis=0)

for n_layer in range(n_layers):

    fig, ax = plt.subplots()

    ax.plot(np.arange(0, magnitude_max+magnitude_step, magnitude_step), results_mean[n_layer, :, metric])

    ax.set_xlabel("Magnitude")
    ax.set_ylabel(f"{metric_names[metric]}")
    ax.set_title(f"Mean results | Layer {n_layer}")

    fig1, ax1 = plt.subplots()

    for i in range(results.shape[2]):
        ax1.scatter([i] * n_reps, results[:, n_layer,  i, metric])

    ax1.set_xticklabels([str(x) for x in range(-1, int(magnitude_max+magnitude_step*10), 1)])

    ax1.set_xlabel("Magnitude")
    ax1.set_ylabel(f"{metric_names[metric]}")
    ax1.set_title(f"Results | Layer {n_layer}")

colors = ["red", "blue", "green", "orange"]

fig2, ax2 = plt.subplots()

for n_m, n_magnitude in enumerate(magnitudes):
    for i in range(n_layers):
        ax2.scatter([n_m]*n_reps, results[:, i, n_m, metric], color=colors[i], label=f"Layer {i}" if n_m == 0 else None)

ax2.set_xticklabels([str(x) for x in range(-1, int(magnitude_max+magnitude_step*10), 1)])

ax2.set_xlabel("Magnitude")
ax2.set_ylabel(f"{metric_names[metric]}")
ax2.set_title("Results")
ax2.legend()

# Moze przezroczystosc
# fig2.savefig("experiment1_layers_all_values.png")

plt.show()