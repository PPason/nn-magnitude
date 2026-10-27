import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# 0 = mean abs
# 1 = norm
metric = 1

chunks = [50, 100, 200, 400, 700, 1000]

n_layers = 4

# ((n_reps, len(chunks), 2))
# # ((20, 6, 2))
results_magnitude = np.load("results_magnitude.npy")
results_magnitude_mean = np.mean(results_magnitude, axis=0)
results_magnitude_std = np.std(results_magnitude, axis=0)

# ((n_reps, n_layers, len(chunks), 2))
# ((20, 4, 6, 2))
results_params = np.load("results_params.npy")

# ((n_reps, n_layers, len(chunks), 2))
baseline = np.load("baselines.npy")

# ((4, 6, 2))
baseline = np.mean(baseline, axis=0)

layer = 0

how_much_more = []

for i, chunk in enumerate(chunks):
    how_much_more.append(np.expand_dims(results_params[:, :, i, :]/baseline[:, i, :], axis=2))

how_much_more = np.concat(how_much_more, axis=2)
how_much_more_mean = np.mean(how_much_more, axis=0)
how_much_more_std = np.std(how_much_more, axis=0)

print("RELATIVE")

for i, chunk in enumerate(chunks):
    print(f"Magnitude: {results_magnitude_mean[i][0]:.3f}")

    for n_layer in range(n_layers):
        print(f"Layer {n_layer} | {how_much_more_mean[n_layer, i, metric]:.3f} ({how_much_more_std[n_layer, i, metric]:.3f})")

print("\nABSOLUTE")

results_params_mean = np.mean(results_params, axis=0)
results_params_std = np.std(results_params, axis=0)

for i, chunk in enumerate(chunks):
    print(f"Magnitude: {results_magnitude_mean[i][0]:.3f}")

    for n_layer in range(n_layers):
        print(f"Layer {n_layer} | {results_params_mean[n_layer, i, metric]:.3f} ({results_params_std [n_layer, i, metric]:.3f})")

alpha = 0.05

# (20, 4, 6, 2)
# print(how_much_more.shape)

for n_ch, n_chunk in enumerate(chunks):
    res = friedmanchisquare(how_much_more[:, 0, n_ch, metric], how_much_more[:, 1, n_ch, metric], how_much_more[:, 2, n_ch, metric], how_much_more[:, 3, n_ch, metric])

    print(f"Magnitude: {results_magnitude_mean[n_ch][0]:.3f}")

    if res.pvalue < alpha:
        res = sp.posthoc_nemenyi_friedman(how_much_more[:, :, n_ch, metric]).to_numpy()

        for i in range(n_layers):
            for j in range(i, n_layers):
                if res[i, j] < alpha and res[j, i] < alpha:
                    if how_much_more_mean[i, n_ch, metric] > how_much_more_mean[j, n_ch, metric]:
                        print(
                            f"Layer {i} ({how_much_more_mean[i, n_ch, metric]:.3f}) istotnie lepszy od Layer {j} ({how_much_more_mean[j, n_ch, metric]:.3f}) | p = {res[i, j]}")
                    else:
                        print(
                            f"Layer {j} ({how_much_more_mean[j, n_ch, metric]:.3f}) istotnie lepszy od Layer {i} ({how_much_more_mean[i, n_ch, metric]:.3f}) | p = {res[i, j]}")

print("\nABSOLUTE")

for n_ch, n_chunk in enumerate(chunks):
    res = friedmanchisquare(results_params[:, 0, n_ch, metric], results_params[:, 1, n_ch, metric], results_params[:, 2, n_ch, metric], results_params[:, 3, n_ch, metric])

    print(f"Magnitude: {results_magnitude_mean[n_ch][0]:.3f}")

    if res.pvalue < alpha:
        res = sp.posthoc_nemenyi_friedman(results_params[:, :, n_ch, metric]).to_numpy()

        for i in range(n_layers):
            for j in range(i, n_layers):
                if res[i, j] < alpha and res[j, i] < alpha:
                    if results_params_mean[i, n_ch, metric] > results_params_mean[j, n_ch, metric]:
                        print(
                            f"Layer {i} ({results_params_mean[i, n_ch, metric]:.3f}) istotnie lepszy od Layer {j} ({results_params_mean[j, n_ch, metric]:.3f}) | p = {res[i, j]}")
                    else:
                        print(
                            f"Layer {j} ({results_params_mean[j, n_ch, metric]:.3f}) istotnie lepszy od Layer {i} ({results_params_mean[i, n_ch, metric]:.3f}) | p = {res[i, j]}")