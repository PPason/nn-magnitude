import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

# 0 = mean abs
# 1 = norm

metric = 1

# ((50, 2, 5))
results_max_performance_loss = np.load("results_max_performance_loss.npy")

mean_results_max_performance_loss = np.mean(results_max_performance_loss, axis=0)

# for i in range(5):
#     print(f"y noise: {mean_results_max_performance_loss[0, i]}")
#     print(f"x noise: {mean_results_max_performance_loss[1, i]}")

# ((50, 4, 2, 2, 2))
results = np.load("results.npy")

mean_results = np.mean(results, axis=0)

n_layers = 4

for noise_factor in range(5):
    fig, ax = plt.subplots(nrows=2, ncols=2)

    fig.suptitle(f"Mean max performance losses | x noise : {mean_results_max_performance_loss[1, noise_factor]:.3f} | y noise: {mean_results_max_performance_loss[0, noise_factor]:.3f}")

    col = 0
    row = 0

    for n_layer in range(n_layers):
        y_true = np.concatenate([np.zeros(len(results[:, n_layer, metric, 0, noise_factor])), np.ones(len(results[:, n_layer, metric, 1, noise_factor]))])
        scores = np.concatenate([results[:, n_layer, metric, 0, noise_factor], results[:, n_layer, metric, 1, noise_factor]])
        auc_raw = roc_auc_score(y_true, scores)
        auc_separation = max(auc_raw, 1 - auc_raw)

        ax[row, col].scatter(np.arange(len(results[:, n_layer, metric, 0, noise_factor])), results[:, n_layer, metric, 0, noise_factor], label="y noise")
        ax[row, col].scatter(np.arange(len(results[:, n_layer, metric, 1, noise_factor])), results[:, n_layer, metric, 1, noise_factor], label="x noise")
        ax[row, col].set_title(f"Layer {n_layer} | AUC-ROC {auc_separation:.3f}")
        ax[row, col].legend()

        row += 1

        if row == 2:
            row = 0
            col += 1

for noise_factor in range(5):
    y_true_max_performance_loss = np.concatenate([np.zeros(len(results_max_performance_loss[:, 0, noise_factor])), np.ones(len(results_max_performance_loss[:, 1, noise_factor]))])
    scores_max_performance_loss = np.concatenate([results_max_performance_loss[:, 0, noise_factor], results_max_performance_loss[:, 1, noise_factor]])
    auc_raw_max_performance_loss = roc_auc_score(y_true_max_performance_loss, scores_max_performance_loss)
    auc_separation_max_performance_loss = max(auc_raw_max_performance_loss, 1 - auc_raw_max_performance_loss)

    print(f"Mean max performance loss | y noise {mean_results_max_performance_loss[0, noise_factor]:.3f} | x noise {mean_results_max_performance_loss[1, noise_factor]:.3f} | AUC-ROC {auc_separation_max_performance_loss:.3f}")

    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(results_max_performance_loss[:, 0, noise_factor])), results_max_performance_loss[:, 0, noise_factor], label=f"y noise")
    ax.scatter(np.arange(len(results_max_performance_loss[:, 1, noise_factor])), results_max_performance_loss[:, 1, noise_factor], label=f"x noise")
    ax.set_title(f"y noise {mean_results_max_performance_loss[0, noise_factor]:.3f} | drift {mean_results_max_performance_loss[1, noise_factor]:.3f} | AUC-ROC {auc_separation_max_performance_loss:.3f}")
    ax.legend()

plt.show()