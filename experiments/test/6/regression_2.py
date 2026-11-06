import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

# 0 = mean abs
# 1 = norm

metric = 1

# ((100, 4, 2, 2))
# n_reps x n_layers x metric x type
# type = y noise/drift
results = np.load("results_max_performance_loss.npy")

y = np.concat([np.zeros(100), np.ones(100)])
X = results.T.flatten().reshape(-1, 1)

skf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

results = np.zeros(10)

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
    results[i] = accuracy_score(y_test, predictions)

print(f"{np.mean(results[:]):.3f} ({np.std(results[:]):.3f})")