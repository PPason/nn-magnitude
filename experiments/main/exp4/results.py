import numpy as np
from sklearn.metrics import roc_auc_score

# 0 = mean abs
# 1 = norm

metric = 1

n_chunks = [150, 250, 400, 600, 1000]

results_max_performance_loss = np.load("results_max_performance_loss.npy")

mean_results_max_performance_loss = np.mean(results_max_performance_loss, axis=0)

print("mean max performance losses")

for i, n_chunk in enumerate(n_chunks):
    print(n_chunk)
    print(f"Noise: {mean_results_max_performance_loss[i, 0]}")
    print(f"Drift: {mean_results_max_performance_loss[i, 1]}")

# n_reps x n_chunks x n_layers x metric x variant (noise/drift)
# ((100, 6, 4, 2, 2))
results = np.load("results.npy")

mean_results = np.mean(results, axis=0)

n_layers = 4

for i, n_chunk in enumerate(n_chunks):
    for n_layer in range(n_layers):
        y_true = np.concatenate([np.zeros(len(results[:, i, n_layer, metric, 0])), np.ones(len(results[:, i, n_layer, metric, 1]))])
        scores = np.concatenate([results[:, i, n_layer, metric, 0], results[:, i, n_layer, metric, 1]])

        auc_raw = roc_auc_score(y_true, scores)
        auc_separation = max(auc_raw, 1 - auc_raw)

        print(f"Layer {n_layer}: AUC-ROC = {auc_separation:.3f}")

    y_true_max_performance_loss = np.concatenate([np.zeros(len(results_max_performance_loss[:, 0])), np.ones(len(results_max_performance_loss[:, 1]))])
    scores_max_performance_loss = np.concatenate([results_max_performance_loss[:, i, 0], results_max_performance_loss[:, i, 1]])
    auc_raw_max_performance_loss = roc_auc_score(y_true_max_performance_loss, scores_max_performance_loss)
    auc_separation_max_performance_loss = max(auc_raw_max_performance_loss, 1 - auc_raw_max_performance_loss)

    print(f"AUC-ROC max performance loss = {auc_separation_max_performance_loss:.3f}")