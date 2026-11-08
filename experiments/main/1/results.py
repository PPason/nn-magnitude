import numpy as np
from scipy import stats

# 0 = mean abs
# 1 = norm
metric = 1
metric_names = ["mean abs", "relative norm change"]

results = np.load("results.npy")
results_layers = np.load("results_layers.npy")

magnitude_max = 5
magnitude_step = 0.1

magnitudes = np.linspace(0, magnitude_max, int(round(magnitude_max / magnitude_step)) + 1)

n_reps = 50

maxes = []

for i in range(results_layers.shape[2]):
    for j in range(results_layers.shape[0]):
        maxes.append(np.argmax(results_layers[j, :, i, metric]))

print(np.unique(maxes, return_counts=True))

results = results[:, :, metric].T.flatten()

magnitudes_list = [x for x in magnitudes for y in range(n_reps)]

res = stats.spearmanr(magnitudes_list, results)

print("Whole net")

print("SPEARMAN")
print(f"{metric_names[metric]} | t: {res.statistic}, p: {res.pvalue}")

res = stats.pearsonr(magnitudes_list, results)

print("PEARSON")
print(f"{metric_names[metric]} | t: {res.statistic}, p: {res.pvalue}")

n_layers = 4

for n_layer in range(n_layers):
    print(f"Layer {n_layer}")

    results = results_layers[:, n_layer, :, metric].T.flatten()

    res = stats.spearmanr(magnitudes_list, results)

    print("SPEARMAN")
    print(f"{metric_names[metric]} | t: {res.statistic}, p: {res.pvalue}")

    res = stats.pearsonr(magnitudes_list, results)

    print("PEARSON")
    print(f"{metric_names[metric]} | t: {res.statistic}, p: {res.pvalue}")