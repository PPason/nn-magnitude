import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

# 0 = mean abs
# 1 = norm
metric_params = 1

metric_params_names = ["abs mean", "relative norm"]

# 0 = max performance loss
# 1 = duration
metric_magnitude = 0

metric_magnitude_names = ["max performance loss", "duration"]

chunks = [50, 100, 200, 400, 700, 1000]

results_magnitude = np.load("results_magnitude.npy")
results_params = np.load("results_params.npy")

magnitudes = []

for i in range(len(chunks)):
    print(f"Chunks {chunks[i]} | {np.mean(results_magnitude[:, i, metric_magnitude]):.3f} ({(np.std(results_magnitude[:, i, metric_magnitude])):.3f})")
    magnitudes.append(np.mean(results_magnitude[:, i, metric_magnitude]))

results_params_abs = results_params[:, :, 0].T.flatten()
results_params_norm = results_params[:, :, 1].T.flatten()

results_magnitudes = results_magnitude[:, :, metric_magnitude].T.flatten()

res1 = stats.spearmanr(results_magnitudes, results_params_abs)
res2 = stats.spearmanr(results_magnitudes, results_params_norm )

print("SPEARMAN")
print(f"abs | t: {res1.statistic}, p: {res1.pvalue}")
print(f"norm| t: {res2.statistic}, p: {res2.pvalue}")

res1 = stats.pearsonr(results_magnitudes, results_params_abs)
res2 = stats.pearsonr(results_magnitudes, results_params_norm)

print("PEARSON")
print(f"abs | t: {res1.statistic}, p: {res1.pvalue}")
print(f"norm| t: {res2.statistic}, p: {res2.pvalue}")

fig, ax = plt.subplots()

ax.scatter(magnitudes, np.mean(results_params[:, :, metric_params], axis=0))
ax.set_ylabel(f"{metric_params_names[metric_params]}")
ax.set_xlabel(f"Magnitudes ({metric_magnitude_names[metric_magnitude]})")
ax.set_title("Mean results")

fig1, ax1 = plt.subplots()

for i in range(results_params.shape[1]):
    ax1.scatter([i] * 20, results_params[:, i, metric_params])

magnitudes.insert(0, 0)

ax1.set_xticklabels(np.round(magnitudes, 3))

ax1.set_xlabel(f"Magnitude ({metric_magnitude_names[metric_magnitude]})")
ax1.set_ylabel(f"{metric_params_names[metric_params]}")
ax1.set_title("Results")

plt.show()