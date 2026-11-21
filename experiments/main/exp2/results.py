import numpy as np
from scipy import stats

# 0 = mean abs
# 1 = norm
metric_params = 1
metric_params_names = ["mean abs", "relative norm change"]

# 0 = max performance loss
# 1 = duration
metric_magnitude = 1

results_params_layers = np.load("results_params.npy")
results_params_net = np.load("results_params_net.npy")
results_magnitudes = np.load("results_magnitude.npy")

chunks = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

n_layers = 4

results_magnitudes_mean = np.mean(results_magnitudes, axis=0)
results_magnitudes_std = np.std(results_magnitudes, axis=0)

for i, n_chunk in enumerate(chunks):
    print(f"{n_chunk} | {results_magnitudes_mean[i, metric_magnitude]:.3f} ({results_magnitudes_std[i, metric_magnitude]:.3f})")

results_params = results_params_net[:, :, metric_params].T.flatten()

results_magnitudes = results_magnitudes[:, :, metric_magnitude].T.flatten()

res = stats.spearmanr(results_magnitudes, results_params)

print("Whole net")

print("SPEARMAN")
print(f"{metric_params_names[metric_params]} | t: {res.statistic}, p: {res.pvalue}")

res = stats.pearsonr(results_magnitudes, results_params)

print("PEARSON")
print(f"{metric_params_names[metric_params]} | t: {res.statistic}, p: {res.pvalue}")

for n_layer in range(n_layers):
    print(f"layer {n_layer}")

    results_params = results_params_layers[:, n_layer, :, metric_params].T.flatten()

    res = stats.spearmanr(results_magnitudes, results_params)

    print("SPEARMAN")
    print(f"{metric_params_names[metric_params]} | t: {res.statistic}, p: {res.pvalue}")

    res = stats.pearsonr(results_magnitudes, results_params)

    print("PEARSON")
    print(f"{metric_params_names[metric_params]} | t: {res.statistic}, p: {res.pvalue}")