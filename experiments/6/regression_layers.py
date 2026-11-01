import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

# 0 = mean abs
# 1 = norm

metric = 1

n_layers = 4

results = np.zeros((n_layers, 10))

for n_layer in range(n_layers):
    # ((100, 4, 2, 2))
    # n_reps x n_layers x metric x type
    # type = y noise/drift
    results_exp = np.load("results.npy")

    y = np.concat([np.zeros(results_exp.shape[0]), np.ones(results_exp.shape[0])])
    X = np.concatenate([results_exp[:, n_layer, metric, 0], results_exp[:, n_layer, metric, 1]])

    skf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X[train_index].reshape(-1, 1)
        y_train = y[train_index]
        X_test = X[test_index].reshape(-1, 1)
        y_test = y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        cls = LogisticRegression(max_iter=100)

        predictions = cls.fit(X_train, y_train).predict(X_test)

        # print(f"Fold: {i} | {accuracy_score(y_test, predictions):.3f}")
        results[n_layer, i] = accuracy_score(y_test, predictions)

for n_layer in range(n_layers):
    print(f"Only {n_layer} layer | {np.mean(results[n_layer, :]):.3f} ({np.std(results[n_layer, :]):.3f})")