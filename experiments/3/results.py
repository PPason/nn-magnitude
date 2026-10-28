import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# 0 = mean abs
# 1 = norm
metric = 1

metric_names = ["abs mean", "relative norm"]

magnitudes = [0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]

scaler = 1000

n_layers = 4

# 20 x 4 x 7 x 2
# results = np.zeros((n_reps, n_layers, len(magnitudes), 2))

results = np.load("results.npy")

# 4 x 2
baseline = np.mean(results[:, :, 0, :], axis=0)

how_much_more = []

for i, magnitude in enumerate(magnitudes[1:]):
    how_much_more.append(np.expand_dims(results[:, :, i, :]/baseline, axis=2))

# 20 x 4 x 6 x 2
how_much_more = np.concat(how_much_more, axis=2)
how_much_more_mean = np.mean(how_much_more, axis=0)
how_much_more_std = np.std(how_much_more, axis=0)

print("RELATIVE")

for i, magnitude in enumerate(magnitudes[1:]):
    print(magnitude)

    for n_layer in range(n_layers):
        print(f"Layer {n_layer} | {how_much_more_mean[n_layer, i, metric]:.3f} ({how_much_more_std[n_layer, i, metric]:.3f})")

print("\nABSOLUTE [10^-3]")

results_mean = np.mean(results, axis=0)
results_std = np.std(results, axis=0)

for i, magnitude in enumerate(magnitudes):
    print(magnitude)

    for n_layer in range(n_layers):
        print(f"Layer {n_layer} | {results_mean[n_layer, i, metric]*scaler:.3f} ({results_std[n_layer, i, metric]*scaler:.3f})")

alpha = 0.05

print("RELATIVE")

for n_ch, n_chunk in enumerate(magnitudes[1:]):
    res = friedmanchisquare(how_much_more[:, 0, n_ch, metric], how_much_more[:, 1, n_ch, metric], how_much_more[:, 2, n_ch, metric], how_much_more[:, 3, n_ch, metric])

    print(n_chunk)

    if res.pvalue < alpha:
        res = sp.posthoc_nemenyi_friedman(how_much_more[:, :, n_ch, metric]).to_numpy()

        for i in range(n_layers):
            for j in range(i, n_layers):
                if res[i, j] < alpha:
                    if how_much_more_mean[i, n_ch, metric] > how_much_more_mean[j, n_ch, metric]:
                        print(
                            f"Layer {i} ({how_much_more_mean[i, n_ch, metric]:.3f}) istotnie lepszy od Layer {j} ({how_much_more_mean[j, n_ch, metric]:.3f}) | p = {res[i, j]}")
                    else:
                        print(
                            f"Layer {j} ({how_much_more_mean[j, n_ch, metric]:.3f}) istotnie lepszy od Layer {i} ({how_much_more_mean[i, n_ch, metric]:.3f}) | p = {res[i, j]}")

print("\nABSOLUTE [10^-3]")

for n_ch, n_chunk in enumerate(magnitudes):
    res = friedmanchisquare(results[:, 0, n_ch, metric], results[:, 1, n_ch, metric], results[:, 2, n_ch, metric], results[:, 3, n_ch, metric])

    print(n_chunk)

    if res.pvalue < alpha:
        res = sp.posthoc_nemenyi_friedman(results[:, :, n_ch, metric]).to_numpy()

        for i in range(n_layers):
            for j in range(i, n_layers):
                if res[i, j] < alpha and res[j, i] < alpha:
                    if results_mean[i, n_ch, metric] > results_mean[j, n_ch, metric]:
                        print(
                            f"Layer {i} ({results_mean[i, n_ch, metric]*scaler:.3f}) istotnie lepszy od Layer {j} ({results_mean[j, n_ch, metric]*scaler:.3f}) | p = {res[i, j]}")
                    else:
                        print(
                            f"Layer {j} ({results_mean[j, n_ch, metric]*scaler:.3f}) istotnie lepszy od Layer {i} ({results_mean[i, n_ch, metric]*scaler:.3f}) | p = {res[i, j]}")

fig, ax = plt.subplots()

colors = ["red", "blue", "green", "orange"]

for n_ch, n_chunk in enumerate(magnitudes):
    for i in range(results_mean.shape[0]):
        ax.scatter(n_ch, results_mean[i, n_ch, metric], color=colors[i], label=f"Layer {i}" if n_ch == 0 else None)

magnitudes.insert(0, 0)

ax.set_xticklabels(magnitudes)

ax.set_xlabel("Magnitude")
ax.set_ylabel(f"{metric_names[metric]}")
ax.set_title("Mean results")

ax.legend()

fig1, ax1 = plt.subplots()

offsets = (np.arange(n_layers) - (n_layers - 1) / 2) * 0.8 / n_layers

for n_ch, n_chunk in enumerate(magnitudes[1:]):
    for i in range(results.shape[1]):
        ax1.scatter([n_ch + offsets[i]] * 20, results[:, i, n_ch, metric], color=colors[i], label=f"Layer {i}" if n_ch == 0 else None)

ax1.set_xticklabels(magnitudes)

ax1.set_xlabel("Magnitude")
ax1.set_ylabel(f"{metric_names[metric]}")
ax1.set_title("Results")

ax1.legend()

plt.show()