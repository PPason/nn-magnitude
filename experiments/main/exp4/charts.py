import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

# 0 = mean abs
# 1 = norm

metric = 1

n_chunks = [150, 250, 400, 600, 1000]

results_max_performance_loss = np.load("results_max_performance_loss.npy")

mean_results_max_performance_loss = np.mean(results_max_performance_loss, axis=0)

# n_reps x n_chunks x n_layers x metric x variant (noise/drift)
# ((100, 5, 4, 2, 2))
results = np.load("results.npy")

n_layers = 4

for i, n_chunk in enumerate(n_chunks):

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout()

    col = 0
    row = 0

    fig.suptitle(f"average max performance loss | drift : {mean_results_max_performance_loss[i, 1]:.3f} | y: {mean_results_max_performance_loss[i, 0]:.3f}")

    for n_layer in range(n_layers):
        y_true = np.concatenate([np.zeros(len(results[:, i, n_layer, metric, 0])), np.ones(len(results[:, i, n_layer, metric, 1]))])
        scores = np.concatenate([results[:, i, n_layer, metric, 0], results[:, i, n_layer, metric, 1]])

        auc_raw = roc_auc_score(y_true, scores)
        auc_separation = max(auc_raw, 1 - auc_raw)

        ax[row, col].scatter(np.arange(len(results[:, i, n_layer, metric, 0])), results[:, i, n_layer, metric, 0], label="y noise")
        ax[row, col].scatter(np.arange(len(results[:, i, n_layer, metric, 1])), results[:, i, n_layer, metric, 1], label="drift")
        ax[row, col].set_title(f"Layer {n_layer} | AUC-ROC {auc_separation:.3f}")
        ax[row, col].grid()
        ax[row, col].spines["top"].set_visible(0)
        ax[row, col].spines["right"].set_visible(0)
        ax[row, col].spines["left"].set_visible(0)
        ax[row, col].spines["bottom"].set_visible(0)

        if row == 0 and col == 0:
            ax[row, col].legend()

        row += 1

        if row == 2:
            row = 0
            col += 1

    fig1, ax1 = plt.subplots()
    fig1.tight_layout()

    y_true_max_performance_loss = np.concatenate([np.zeros(len(results_max_performance_loss[:, 0])), np.ones(len(results_max_performance_loss[:, 1]))])
    scores_max_performance_loss = np.concatenate([results_max_performance_loss[:, i, 0], results_max_performance_loss[:, i, 1]])
    auc_raw_max_performance_loss = roc_auc_score(y_true_max_performance_loss, scores_max_performance_loss)
    auc_separation_max_performance_loss = max(auc_raw_max_performance_loss, 1 - auc_raw_max_performance_loss)

    ax1.scatter(np.arange(len(results_max_performance_loss[:, i, 0])), results_max_performance_loss[:, i, 0], label=f"y noise")
    ax1.scatter(np.arange(len(results_max_performance_loss[:, i, 1])), results_max_performance_loss[:, i, 1], label=f"drift")
    ax1.set_title(f"Max performance losses: AUC-ROC {auc_separation_max_performance_loss:.3f}")
    ax1.legend()
    ax1.grid()
    ax1.spines["top"].set_visible(0)
    ax1.spines["right"].set_visible(0)
    ax1.spines["left"].set_visible(0)
    ax1.spines["bottom"].set_visible(0)

    # if i == 1:
    #     fig.savefig("experiment3_parameters.png")
    #     fig1.savefig("experiment3_losses.png")

plt.show()