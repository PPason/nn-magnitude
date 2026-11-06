import numpy as np
from scipy import stats

magnitudes = [0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
n_reps = 20

results = np.load("results.npy")

# print(results.shape)

results_abs = results[:, :, 0].T.flatten()
results_norm = results[:, :, 1].T.flatten()

magnitudes_list = [x for x in magnitudes for y in range(n_reps)]

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