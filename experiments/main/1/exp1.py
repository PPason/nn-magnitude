import numpy as np
from sklearn.neural_network import MLPClassifier
from generator.OWDSG import OWDSG

magnitude_max = 5
magnitude_step = 0.1

magnitudes = np.linspace(0, magnitude_max, int(round(magnitude_max / magnitude_step)) + 1)

n_reps = 50

n_layers = 4

results = np.zeros((n_reps, len(magnitudes), 2))
results_layers = np.zeros((n_reps, n_layers, len(magnitudes), 2))
random_states = np.zeros((n_reps * len(magnitudes)))

n_chunks = 300
batch_size = 100
n_features = 100
n_informative_percent = 0.8
n_classes = 10

for n_m, magnitude in enumerate(magnitudes):
    print('n_m:', n_m)

    for n_rep in range(n_reps):
        random_state = np.random.randint(0, 42949672)
        random_states[n_m * n_reps + n_rep] = random_state

        stream = OWDSG(n_drifts=1,
                       n_novel=0,
                       n_classes=n_classes,
                       n_features=n_features,
                       n_informative=int(n_features * n_informative_percent),
                       n_chunks=n_chunks,
                       chunk_size=batch_size,
                       drift_magnitude=magnitude,
                       smooth_drift=True,
                       random_state=random_state)

        mlp = MLPClassifier(hidden_layer_sizes=(40, 40, 40, ), batch_size=batch_size, random_state=random_state)

        first = True

        params = None

        params_layers = []

        for n_chunk in range(n_chunks):
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

                for n_layer in range(mlp.n_layers_ - 1):
                    params_layers.append(np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()]))
                    params_layers[n_layer] = np.expand_dims(params_layers[n_layer], axis=0)

                continue

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

            for n_layer in range(mlp.n_layers_ - 1):
                params_layers[n_layer] = np.concat([params_layers[n_layer], np.expand_dims(
                    np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()]), axis=0)])

        for n_layer in range(mlp.n_layers_ - 1):
            params_diff_abs = np.mean(np.abs(np.diff(params_layers[n_layer], axis=0)), axis=1)
            params_diff_norm = np.linalg.norm(np.diff(params_layers[n_layer], axis=0), axis=1) / np.linalg.norm(params_layers[n_layer][:-1], axis=1)

            results_layers[n_rep, n_layer, n_m, 0] = np.max(params_diff_abs[int(n_chunks/2):])
            results_layers[n_rep, n_layer, n_m, 1] = np.max(params_diff_norm[int(n_chunks/2):])

        params_diff_mean = np.mean(np.abs(np.diff(params, axis=0)), axis=1)

        params_diff_norm = np.linalg.norm(np.diff(params, axis=0), axis=1)/np.linalg.norm(params[:-1], axis=1)

        results[n_rep, n_m, 0] = np.max(params_diff_mean[int(n_chunks/2):])
        results[n_rep, n_m, 1] = np.max(params_diff_norm[int(n_chunks/2):])

np.save("results.npy", results)
np.save("random_states.npy", random_states)
np.save("results_layers.npy", results_layers)