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

n_layers = 4

random_states_mlp = np.zeros((n_reps * len(chunks)))
random_states_stream = np.zeros((n_reps * len(chunks)))

results_magnitude = np.zeros((n_reps, len(chunks), 2))
results_params = np.zeros((n_reps, n_layers, len(chunks), 2))

for n_ch, chunk in enumerate(chunks):
    print(f"chunk {chunk}")

    for n_rep in range(n_reps):
        print(f"rep {n_rep}")

        stream = OWDSG(n_drifts=1,
                       n_novel=0,
                       n_classes=n_classes,
                       n_features=n_features,
                       n_informative=int(n_features * n_informative_percent),
                       n_chunks=chunk,
                       chunk_size=batch_size,
                       smooth_drift=False)

        mlp = MLPClassifier(hidden_layer_sizes=(40, 40, 40, ), batch_size=batch_size)

        accuracies = []

        first = True

        params = []

        for n_chunk in range(chunk):
            X, y = stream.get_chunk(n_chunk)

            if first:
                mlp.partial_fit(X, y, classes=np.arange(n_classes))

                for n_layer in range(mlp.n_layers_ - 1):
                    params.append(np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()]))
                    params[n_layer] = np.expand_dims(params[n_layer], axis=0)

                first = False

                continue

            acc = accuracy_score(y, mlp.predict(X))
            accuracies.append(acc)

            mlp.partial_fit(X, y, classes=np.arange(n_classes))

            for n_layer in range(mlp.n_layers_ - 1):
                params[n_layer] = np.concat([params[n_layer], np.expand_dims(np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()]), axis=0)])

        for n_layer in range(mlp.n_layers_ - 1):
            params_diff_abs = np.mean(np.abs(np.diff(params[n_layer], axis=0)), axis=1)
            params_diff_norm = np.linalg.norm(np.diff(params[n_layer], axis=0), axis=1) / np.linalg.norm(params[n_layer][:-1], axis=1)

            results_params[n_rep, n_layer, n_ch, 0] = np.max(params_diff_abs[int(chunk / 2) - 10:])
            results_params[n_rep, n_layer, n_ch, 1] = np.max(params_diff_norm[int(chunk / 2) - 10:])

        results_magnitude[n_rep, n_ch, 0] = max_performance_loss(acc=np.array(accuracies),drift=stream.get_drift_gt()[0])
        results_magnitude[n_rep, n_ch, 1] = duration(acc=np.array(accuracies), drift=stream.get_drift_gt()[0])

        random_states_mlp[n_ch * n_reps + n_rep] = mlp.random_state
        random_states_stream[n_ch * n_reps + n_rep] = stream.random_state

baselines = np.zeros((n_reps, n_layers, len(chunks), 2))

print("baseline")

for n_ch, chunk in enumerate(chunks):
    print(f"chunk {chunk}")

    for n_rep in range(n_reps):
        print(f"rep {n_rep}")

        stream = OWDSG(n_drifts=0,
                       n_novel=0,
                       n_classes=n_classes,
                       n_features=n_features,
                       n_informative=int(n_features * n_informative_percent),
                       n_chunks=chunk,
                       chunk_size=batch_size,
                       smooth_drift=False)

        mlp = MLPClassifier(hidden_layer_sizes=(40, 40, 40, ), batch_size=batch_size)

        accuracies = []

        first = True

        params = []

        for n_chunk in range(chunk):
            X, y = stream.get_chunk(n_chunk)

            if first:
                mlp.partial_fit(X, y, classes=np.arange(n_classes))

                for n_layer in range(mlp.n_layers_ - 1):
                    params.append(np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()]))
                    params[n_layer] = np.expand_dims(params[n_layer], axis=0)

                first = False

                continue

            acc = accuracy_score(y, mlp.predict(X))
            accuracies.append(acc)

            mlp.partial_fit(X, y, classes=np.arange(n_classes))

            for n_layer in range(mlp.n_layers_ - 1):
                params[n_layer] = np.concat([params[n_layer], np.expand_dims(np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()]), axis=0)])

        for n_layer in range(mlp.n_layers_ - 1):
            params_diff_abs = np.mean(np.abs(np.diff(params[n_layer], axis=0)), axis=1)
            params_diff_norm = np.linalg.norm(np.diff(params[n_layer], axis=0), axis=1) / np.linalg.norm(
                params[n_layer][:-1], axis=1)

            baselines[n_rep, n_layer, n_ch, 0] = np.max(params_diff_abs[int(chunk / 2) - 10:])
            baselines[n_rep, n_layer, n_ch, 1] = np.max(params_diff_norm[int(chunk / 2) - 10:])

np.save("results_magnitude.npy", results_magnitude)
np.save("results_params.npy", results_params)
np.save("random_states_mlp.npy", random_states_mlp)
np.save("random_states_stream.npy", random_states_stream)
np.save("baselines.npy", baselines)