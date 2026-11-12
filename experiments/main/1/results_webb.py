import numpy as np
from scipy import stats

# 0 = mean abs
# 1 = norm
metric = 1
metric_names = ["mean abs", "relative norm change"]

metric_webb_names = ["DM", "CMCD", "PD"]

# 0 = max
# 1 = mean
mean_or_max = 1

results = np.load("results.npy")
results_layers = np.load("results_layers.npy")
results_webb = np.load("results_webb.npy")

n_reps = 20

results = results[:, :, metric].T.flatten()

for i in range(results_webb.shape[2]):
    print("===========================")
    print(f"{metric_webb_names[i]}")

    res = stats.spearmanr(results_webb[:, :, i, mean_or_max].T.flatten(), results)

    print("Whole net")

    print("SPEARMAN")
    print(f"{metric_names[metric]} | t: {res.statistic}, p: {res.pvalue}")

    res = stats.pearsonr(results_webb[:, :, i, mean_or_max].T.flatten(), results)

    print("PEARSON")
    print(f"{metric_names[metric]} | t: {res.statistic}, p: {res.pvalue}")

    n_layers = 4

    for n_layer in range(n_layers):
        print(f"Layer {n_layer}")

        results = results_layers[:, n_layer, :, metric].T.flatten()

        res = stats.spearmanr(results_webb[:, :, i, mean_or_max].T.flatten(), results)

        print("SPEARMAN")
        print(f"{metric_names[metric]} | t: {res.statistic}, p: {res.pvalue}")

        res = stats.pearsonr(results_webb[:, :, i, mean_or_max].T.flatten(), results)

        print("PEARSON")
        print(f"{metric_names[metric]} | t: {res.statistic}, p: {res.pvalue}")