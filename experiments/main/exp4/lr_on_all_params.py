import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

# 0 = mean abs
# 1 = norm

metric = 1

# ((100, 5, 4, 2, 2))
# n_reps x n_chunks x n_layers x metric x type
# type = y noise/drift
results = np.load("results.npy")

mean_results_max_performance_loss = np.mean(np.load("results_max_performance_loss.npy"), axis=0)

y = np.concat([np.zeros(results.shape[0]), np.ones(results.shape[0])])

n_splits = 2
n_repeats = 5

skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

results_exp = np.zeros((results.shape[1], n_repeats*n_splits))

for n in range(results.shape[1]):
    X = np.concatenate([results[:, n, :, metric, 0], results[:, n, :, metric, 1]])

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        cls = LogisticRegression(max_iter=100)

        predictions = cls.fit(X_train, y_train).predict(X_test)

        # print(f"Fold: {i} | {accuracy_score(y_test, predictions):.3f}")
        results_exp[n, i] = accuracy_score(y_test, predictions)

for n in range(results.shape[1]):
    print(f"Max performance loss | noise {mean_results_max_performance_loss[n, 0]:.3f} | drift {mean_results_max_performance_loss[n, 1]:.3f}")
    print(f"{np.mean(results_exp[n, :]):.3f} ({np.std(results_exp[n, :]):.3f})")