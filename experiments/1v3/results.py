import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

# 0 = abs mean
# 1 = norm
metric = 1

magnitude_max = 5
magnitude_step = 0.1

magnitudes = np.linspace(0, magnitude_max, int(round(magnitude_max / magnitude_step)) + 1)

n_reps = 20

results = np.load("results.npy")

# print(results.shape)

results_abs = results[:, :, metric].T.flatten()
results_norm = results[:, :, metric].T.flatten()

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

plt.show()