import numpy as np
from sklearn.neural_network import MLPClassifier
from generator.OWDSG import OWDSG

magnitude_max = 5
magnitude_step = 0.1

magnitudes = np.linspace(0, magnitude_max, int(round(magnitude_max / magnitude_step)) + 1)

n_reps = 20

results = np.zeros((n_reps, len(magnitudes), 2))
random_states_mlp = np.zeros((n_reps * len(magnitudes)))
random_states_stream = np.zeros((n_reps * len(magnitudes)))

n_chunks = 200
batch_size = 100
n_features = 40
n_informative_percent = 0.8
n_classes = 10

for n_m, magnitude in enumerate(magnitudes):
    print('n_m:', n_m)

    for n_rep in range(n_reps):

        stream = OWDSG(n_drifts=1,
                       n_novel=0,
                       n_classes=n_classes,
                       n_features=n_features,
                       n_informative=int(n_features * n_informative_percent),
                       n_chunks=n_chunks,
                       chunk_size=batch_size,
                       drift_magnitude=magnitude,
                       smooth_drift=True)

        mlp = MLPClassifier(hidden_layer_sizes=(40, 40, 40, ), batch_size=batch_size)

        first = True

        params = None

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

        params_diff_mean = np.mean(np.abs(np.diff(params, axis=0)), axis=1)

        params_diff_norm = np.linalg.norm(np.diff(params, axis=0), axis=1)/np.linalg.norm(params[:-1], axis=1)

        results[n_rep, n_m, 0] = np.max(params_diff_mean[125:200])
        results[n_rep, n_m, 1] = np.max(params_diff_norm[125:200])

        random_states_mlp[n_m * n_reps + n_rep] = mlp.random_state
        random_states_stream[n_m * n_reps + n_rep] = stream.random_state

np.save("results.npy", results)
np.save("random_states_mlp.npy", random_states_mlp)
np.save("random_states_stream.npy", random_states_stream)