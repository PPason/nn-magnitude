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
n_chunks = 600

n_reps = 20

n_layers = 4

y_noise = np.arange(0, 1, 0.05)
y_noise_range = [300, 310]

random_states_mlp = np.zeros((n_reps * len(y_noise)))
random_states_stream = np.zeros((n_reps * len(y_noise)))

results_max_performance_loss = np.zeros((n_reps, len(y_noise)))
results_params = np.zeros((n_reps, n_layers, len(y_noise), 2))

for n_f, noise in enumerate(y_noise):
    print(f"y noise {noise}")

    for n_rep in range(n_reps):
        print(f"rep {n_rep}")

        stream = OWDSG(n_drifts=0,
                       n_novel=0,
                       n_classes=n_classes,
                       n_features=n_features,
                       n_informative=int(n_features * n_informative_percent),
                       n_chunks=n_chunks,
                       chunk_size=batch_size,
                       smooth_drift=False)

        mlp = MLPClassifier(hidden_layer_sizes=(40, 40, 40, ), batch_size=batch_size)

        accuracies = []

        first = True

        params = []

        for n_chunk in range(n_chunks):
            X, y = stream.get_chunk(n_chunk)

            if y_noise_range[0] <= n_chunk <= y_noise_range[1]:
                to_shuffle = len(y) * noise
                y_shuffle = np.copy(y)
                np.random.shuffle(y_shuffle)
                y[:int(to_shuffle)] = y_shuffle[:int(to_shuffle)]

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

            results_params[n_rep, n_layer, n_f, 0] = np.max(params_diff_abs[int(n_chunk / 2) - 10:])
            results_params[n_rep, n_layer, n_f, 1] = np.max(params_diff_norm[int(n_chunk / 2) - 10:])

        results_max_performance_loss[n_rep, n_f] = max_performance_loss(acc=np.array(accuracies),drift=int(n_chunk / 2))

        random_states_mlp[n_f * n_reps + n_rep] = mlp.random_state
        random_states_stream[n_f * n_reps + n_rep] = stream.random_state

np.save("results_max_performance_loss.npy", results_max_performance_loss)
np.save("results_params.npy", results_params)
np.save("random_states_mlp.npy", random_states_mlp)
np.save("random_states_stream.npy", random_states_stream)