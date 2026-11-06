import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from generator.OWDSG import OWDSG

def max_performance_loss(acc, drift, window_size=10):
    window = acc[max(0, drift - window_size):drift + window_size]

    return np.max(window) - np.min(window)

def duration(acc, drift, threshold=0.95, window_size=10):
    before = np.mean(acc[max(0, drift - window_size):drift])
    recovered = np.flatnonzero(acc[drift:] >= threshold * before)

    return recovered[0] if len(recovered) else len(acc) - drift

batch_size = 100
n_features = 40
n_informative_percent = 0.8
n_classes = 10

n_reps = 20
chunks = [50, 100, 200, 400, 700, 1000]

results_magnitude = np.zeros((n_reps, len(chunks), 2))
results_params = np.zeros((n_reps, len(chunks), 2))

for n_ch, chunk in enumerate(chunks):

    for n_rep in range(n_reps):

        stream = OWDSG(n_drifts=1,
                       n_novel=0,
                       n_classes=n_classes,
                       n_features=n_features,
                       n_informative=int(n_features * n_informative_percent),
                       n_chunks=chunk,
                       chunk_size=batch_size,
                       smooth_drift=False)

        mlp = MLPClassifier(hidden_layer_sizes=(40,), batch_size=batch_size)

        accuracies = []

        first = True

        params = None

        for n_chunk in range(chunk):
            X, y = stream.get_chunk(n_chunk)

            if first:
                mlp.partial_fit(X, y, classes=np.arange(n_classes))

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

                continue

            acc = accuracy_score(y, mlp.predict(X))
            accuracies.append(acc)

            mlp.partial_fit(X, y, classes=np.arange(n_classes))

            params_net = None

            for n_layer in range(mlp.n_layers_ - 1):
                if params_net is None:
                    params_net = np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()])
                    continue

                params_net = np.concatenate(
                    [params_net, np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()])])

            params_net = np.expand_dims(params_net, axis=0)

            params = np.concat([params, params_net], axis=0)

        params_diff_abs = np.mean(np.abs(np.diff(params, axis=0)), axis=1)
        params_diff_norm = np.linalg.norm(np.diff(params, axis=0), axis=1) / np.linalg.norm(params[:-1], axis=1)

        results_magnitude[n_rep, n_ch, 0] = max_performance_loss(acc=np.array(accuracies), drift=stream.get_drift_gt()[0])
        results_magnitude[n_rep, n_ch, 1] = duration(acc=np.array(accuracies), drift=stream.get_drift_gt()[0])

        results_params[n_rep, n_ch, 0] = np.max(params_diff_abs[int(chunk/2)-10:])
        results_params[n_rep, n_ch, 1] = np.max(params_diff_norm[int(chunk/2)-10:])

np.save("results_magnitude.npy", results_magnitude)
np.save("results_params.npy", results_params)