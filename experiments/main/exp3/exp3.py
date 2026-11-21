import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KernelDensity
from sklearn.neural_network import MLPClassifier
import pandas as panda

def max_performance_loss(acc, drift, window_size=10):
    window = acc[max(0, drift - window_size):drift + window_size]

    return np.max(window) - np.min(window)

def duration(acc, drift, threshold=0.95, window_size=10):
    before = np.mean(acc[max(0, drift - window_size):drift])
    recovered = np.flatnonzero(acc[drift:] >= threshold * before)

    return recovered[0] if len(recovered) else len(acc) - drift

def density(samples, points):
    return np.exp(KernelDensity(bandwidth='scott').fit(samples).score_samples(points))

def dm(x_old, x_new):
    x = np.concatenate([x_old, x_new])
    p_old, p_new = density(x_old, x), density(x_new, x)
    m = (len(x_old) * p_old + len(x_new) * p_new) / len(x)
    return 0.5 * np.mean(np.abs(p_old - p_new) / m)

def cmcd(x_old, y_old, x_new, y_new):
    total = 0.0
    for c in np.unique(y_old):
        weight = (np.mean(y_old == c) + np.mean(y_new == c)) / 2
        total += weight * dm(x_old[y_old == c], x_new[y_new == c])
    return total

def pd(x_old, y_old, x_new, y_new):
    x = np.concatenate([x_old, x_new])
    classes = np.unique(y_old)

    # P(y) p(x|y) dla kazdej klasy w obu okresach
    joint_old, joint_new = {}, {}
    for c in classes:
        joint_old[c] = np.mean(y_old == c) * density(x_old[y_old == c], x)
        joint_new[c] = np.mean(y_new == c) * density(x_new[y_new == c], x)

    # p(x) = sum_y P(y) p(x|y)
    p_old, p_new = np.zeros(len(x)), np.zeros(len(x))
    for c in classes:
        p_old += joint_old[c]
        p_new += joint_new[c]

    # 1/2 sum_y |P_old(y|x) - P_new(y|x)|, gdzie P(y|x) = P(y) p(x|y) / p(x)
    tv = np.zeros(len(x))
    for c in classes:
        tv += 0.5 * np.abs(joint_old[c] / p_old - joint_new[c] / p_new)

    m = (len(x_old) * p_old + len(x_new) * p_new) / len(x)
    weight = (p_old + p_new) / 2 / m
    return np.mean(weight * tv)

magnitudes = [0.0, 0.25, 0.5, 0.75, 1.0]
n_reps = 20

n_layers = 4
n_classes = 2

results = np.zeros((n_reps, len(magnitudes), 2))
results_layers = np.zeros((n_reps, n_layers, len(magnitudes), 2))
results_webb = np.zeros((n_reps, len(magnitudes), 3, 2))
results_shaker = np.zeros((n_reps, len(magnitudes), 2))

for n_magnitude, magnitude in enumerate(magnitudes):
    print(n_magnitude)

    for n_rep in range(n_reps):
        print(f"n_rep: {n_rep}")

        train = panda.read_csv('KDDTrain+.txt', sep=",", header=None)
        test = panda.read_csv('KDDTest+.txt', sep=",", header=None)

        train = train.sample(frac=1)
        test = test.sample(frac=1)

        before = train.copy()

        n_after = len(test)

        n_train = int(n_after * (1 - magnitude))
        n_test = n_after - n_train

        after_train = train.sample(n=n_train, replace=n_train > len(train))

        after_test = test.sample(n=n_test, replace=n_test > len(test))

        after = panda.concat([after_train, after_test])

        after = after.sample(frac=1)

        stream = panda.concat([before, after], ignore_index=True)

        chunk_size = 100
        batch_size = chunk_size

        drift_gt = len(before)
        drift_gt_chunk = drift_gt // chunk_size

        first_post_drift_chunk = (drift_gt + chunk_size - 1) // chunk_size
        drift_accuracy_index = first_post_drift_chunk - 1

        X = stream.drop(columns=[41, 42])
        y = (stream.iloc[:, 41] != 'normal').astype(int)

        X = panda.get_dummies(X, dtype=int)

        X = X.to_numpy()
        y = y.to_numpy()

        saving_range = 20
        save_1_index = 50
        save_2_index = drift_gt_chunk + saving_range
        saved_1 = [[], []]
        saved_2 = [[], []]

        mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10,), batch_size=batch_size)

        accuracies = []

        first = True

        first_X_mean = None
        first_X_std = None

        params = None
        params_layers = []

        for n_chunk in range(int(X.shape[0]/chunk_size)):
            X_batch = X[n_chunk*chunk_size:(n_chunk+1)*chunk_size, :]
            y_batch = y[n_chunk*chunk_size:(n_chunk+1)*chunk_size]

            if save_1_index - saving_range < n_chunk <= save_1_index:
                saved_1[0].append(X_batch)
                saved_1[1].append(y_batch)

            if save_2_index - saving_range < n_chunk <= save_2_index:
                saved_2[0].append(X_batch)
                saved_2[1].append(y_batch)

            if first:
                first_X_mean = np.mean(X_batch, axis=0)
                first_X_std = np.std(X_batch, axis=0)
                first_X_std[first_X_std == 0] = 1

                X_batch = (X_batch - first_X_mean) / first_X_std

                mlp.partial_fit(X_batch, y_batch, classes=np.arange(n_classes))

                params_net = None

                for n_layer in range(mlp.n_layers_ - 1):
                    if params_net is None:
                        params_net = np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()])
                        continue

                    params_net = np.concatenate(
                        [params_net, np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()])])

                params_net = np.expand_dims(params_net, axis=0)

                params = params_net

                first = False

                for n_layer in range(mlp.n_layers_ - 1):
                    params_layers.append(
                        np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()]))
                    params_layers[n_layer] = np.expand_dims(params_layers[n_layer], axis=0)

                continue

            X_batch = (X_batch - first_X_mean)/first_X_std

            predictions = mlp.predict(X_batch)
            accuracies.append(balanced_accuracy_score(y_batch, predictions))

            mlp.partial_fit(X_batch, y_batch, classes=[0, 1])

            params_net = None

            for n_layer in range(mlp.n_layers_ - 1):
                if params_net is None:
                    params_net = np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()])
                    continue

                params_net = np.concatenate(
                    [params_net, np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()])])

            params_net = np.expand_dims(params_net, axis=0)

            params = np.concat([params, params_net], axis=0)

            for n_layer in range(mlp.n_layers_ - 1):
                params_layers[n_layer] = np.concat([params_layers[n_layer], np.expand_dims(
                    np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()]), axis=0)])

        saved_1[0] = np.concatenate(saved_1[0])
        saved_1[1] = np.concatenate(saved_1[1])

        saved_2[0] = np.concatenate(saved_2[0])
        saved_2[1] = np.concatenate(saved_2[1])

        covariate_drifts = []
        conditioned_marginal_covariate_drifts = []
        posterior_drifts = []

        for i in range(X.shape[1]):
            covariate_drifts.append(dm(saved_1[0][:, i].reshape(-1, 1), saved_2[0][:, i].reshape(-1, 1)))
            conditioned_marginal_covariate_drifts.append(cmcd(saved_1[0][:, i].reshape(-1, 1), saved_1[1], saved_2[0][:, i].reshape(-1, 1), saved_2[1]))
            posterior_drifts.append(pd(saved_1[0][:, i].reshape(-1, 1), saved_1[1], saved_2[0][:, i].reshape(-1, 1), saved_2[1]))

        results_webb[n_rep, n_magnitude, 0, 0] = np.nanmax(covariate_drifts)
        results_webb[n_rep, n_magnitude, 1, 0] = np.nanmax(conditioned_marginal_covariate_drifts)
        results_webb[n_rep, n_magnitude, 2, 0] = np.nanmax(posterior_drifts)

        results_webb[n_rep, n_magnitude, 0, 1] = np.nanmean(covariate_drifts)
        results_webb[n_rep, n_magnitude, 1, 1] = np.nanmean(conditioned_marginal_covariate_drifts)
        results_webb[n_rep, n_magnitude, 2, 1] = np.nanmean(posterior_drifts)

        for n_layer in range(mlp.n_layers_ - 1):
            params_diff_abs = np.mean(np.abs(np.diff(params_layers[n_layer], axis=0)), axis=1)
            params_diff_norm = np.linalg.norm(np.diff(params_layers[n_layer], axis=0), axis=1) / np.linalg.norm(params_layers[n_layer][:-1], axis=1)

            results_layers[n_rep, n_layer, n_magnitude, 0] = np.max(params_diff_abs[drift_gt_chunk:drift_gt_chunk+50])
            results_layers[n_rep, n_layer, n_magnitude, 1] = np.max(params_diff_norm[drift_gt_chunk:drift_gt_chunk+50])

        params_diff_mean = np.mean(np.abs(np.diff(params, axis=0)), axis=1)

        params_diff_norm = np.linalg.norm(np.diff(params, axis=0), axis=1)/np.linalg.norm(params[:-1], axis=1)

        results[n_rep, n_magnitude, 0] = np.max(params_diff_mean[drift_gt_chunk:drift_gt_chunk+50])
        results[n_rep, n_magnitude, 1] = np.max(params_diff_norm[drift_gt_chunk:drift_gt_chunk+50])

        results_shaker[n_rep, n_magnitude, 0] = max_performance_loss(acc=np.array(accuracies),drift=drift_accuracy_index)
        results_shaker[n_rep, n_magnitude, 1] = duration(acc=np.array(accuracies), drift=drift_accuracy_index)

np.save("results.npy", results)
np.save("results_layers.npy", results_layers)
np.save("results_webb.npy", results_webb)
np.save("results_shaker.npy", results_shaker)