import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

# 0 = abs mean
# 1 = norm
metric = 1

metric_names = ["abs mean", "relative norm"]

magnitude_max = 5
magnitude_step = 0.1

magnitudes = np.linspace(0, magnitude_max, int(round(magnitude_max / magnitude_step)) + 1)

n_reps = 20

results = np.load("results.npy")

# print(results.shape)

results_abs = results[:, :, 0].T.flatten()
results_norm = results[:, :, 1].T.flatten()

magnitudes_list = [x*magnitude_step for x in magnitudes for y in range(n_reps)]

res1 = stats.spearmanr(magnitudes_list, results_abs)
res2 = stats.spearmanr(magnitudes_list, results_norm)

print("SPEARMAN")
print(f"abs | t: {res1.statistic}, p: {res1.pvalue}")
print(f"norm| t: {res2.statistic}, p: {res2.pvalue}")

res1 = stats.pearsonr(magnitudes_list, results_abs)
res2 = stats.pearsonr(magnitudes_list, results_norm)

print("PEARSON")
print(f"abs | t: {res1.statistic}, p: {res1.pvalue}")
print(f"norm| t: {res2.statistic}, p: {res2.pvalue}")

results_mean = np.mean(results, axis=0)
results_std = np.std(results, axis=0)

fig, ax = plt.subplots()

ax.plot(np.arange(0, magnitude_max+magnitude_step, magnitude_step), results_mean[:, metric])

ax.set_xlabel("Magnitude")
ax.set_ylabel(f"{metric_names[metric]}")
ax.set_title("Mean results")

fig1, ax1 = plt.subplots()

for i in range(results.shape[1]):
    ax1.scatter([i] * 20, results[:, i, metric])

ax1.set_xticklabels([str(x) for x in range(-1, int(magnitude_max+magnitude_step*10), 1)])

ax1.set_xlabel("Magnitude")
ax1.set_ylabel(f"{metric_names[metric]}")
ax1.set_title("Results")

plt.show()