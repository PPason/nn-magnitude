import numpy as np
from matplotlib import pyplot as plt

# 0 = abs mean
# 1 = norm
metric = 1

metric_names = ["abs mean", "relative norm"]

magnitude_max = 5
magnitude_step = 0.1

magnitudes = np.linspace(0, magnitude_max, int(round(magnitude_max / magnitude_step)) + 1)

n_reps = 20

results = np.load("results.npy")

magnitudes_list = [x for x in magnitudes for y in range(n_reps)]

results_mean = np.mean(results, axis=0)
results_std = np.std(results, axis=0)

fig, ax = plt.subplots()
fig.tight_layout()

ax.plot(np.arange(0, magnitude_max+magnitude_step, magnitude_step), results_mean[:, metric])

ax.set_xlabel("Magnitude")
ax.set_ylabel(f"{metric_names[metric]}")
ax.set_title("Mean results")
ax.grid()
ax.spines["top"].set_visible(0)
ax.spines["right"].set_visible(0)
ax.spines["left"].set_visible(0)
ax.spines["bottom"].set_visible(0)

fig1, ax1 = plt.subplots()
fig1.tight_layout()

for i in range(results.shape[1]):
    ax1.scatter([i] * n_reps, results[:, i, metric])

ax1.set_xticklabels([str(x) for x in range(-1, int(magnitude_max+magnitude_step*10), 1)])

ax1.set_xlabel("Magnitude")
ax1.set_ylabel(f"{metric_names[metric]}")
ax1.set_title("Results")
ax1.grid()
ax1.spines["top"].set_visible(0)
ax1.spines["right"].set_visible(0)
ax1.spines["left"].set_visible(0)
ax1.spines["bottom"].set_visible(0)

# fig1.savefig("experiment1_whole_network.png")

plt.show()