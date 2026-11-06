import numpy as np
from scipy import stats

magnitude_max = 5
magnitude_step = 0.1

magnitudes = np.linspace(0, magnitude_max, int(round(magnitude_max / magnitude_step)) + 1)

n_reps = 1

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