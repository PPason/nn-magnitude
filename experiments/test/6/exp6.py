import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from generator.OWDSG import OWDSG

def max_performance_loss(acc, drift, window_size=10):
    window = acc[max(0, drift - window_size):drift + window_size]

    return np.max(window) - np.min(window)

n_chunks = 400
batch_size = 100
n_features = 100
n_informative_percent = 0.8
n_classes = 3

noise_range = [200, 210]
y_noise_factor = 0.95
variants = ["noise", "drift"]

n_reps = 100

n_layers = 4

results = np.zeros((n_reps, n_layers, 2, 2))
results_max_performance_loss = np.zeros((n_reps, 2))

for n_rep in range(n_reps):

    print(n_rep)
    random_state = np.random.randint(0, 42949672)

    for n_v, n_variant in enumerate(variants):

        if n_variant == "noise":

            stream = OWDSG(n_drifts=0,
                               n_novel=0,
                               hide_label=True,
                               n_classes=n_classes,
                               n_features=n_features,
                               n_informative=int(n_features * n_informative_percent),
                               n_chunks=n_chunks,
                               chunk_size=batch_size,
                               drift_magnitude=5,
                               smooth_drift=False,
                               random_state=random_state)

        else:
            stream = OWDSG(n_drifts=1,
                           n_novel=0,
                           hide_label=True,
                           n_classes=n_classes,
                           n_features=n_features,
                           n_informative=int(n_features * n_informative_percent),
                           n_chunks=n_chunks,
                           chunk_size=batch_size,
                           drift_magnitude=5,
                           smooth_drift=False,
                           random_state=random_state)

        mlp = MLPClassifier(hidden_layer_sizes=(40, 40, 40, ), batch_size=batch_size, random_state=random_state)

        accuracies = []

        first = True

        params = []

        for n_chunk in range(n_chunks):
            X, y = stream.get_chunk(n_chunk)

            if n_variant == "noise" and (noise_range[0] <= n_chunk <= noise_range[1]):
                flip_mask = np.random.random(len(y)) < y_noise_factor
                y[flip_mask] = np.random.randint(0, n_classes, size=flip_mask.sum())

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

        results_max_performance_loss[n_rep, n_v] = max_performance_loss(accuracies, noise_range[0])

        params_net = None

        for i in range(params[0].shape[0]):
            if params_net is None:
                params_net =  np.expand_dims(np.concatenate([layer[i, :] for layer in params], axis=0), axis=0)
                continue

            params_net = np.concat([params_net, np.expand_dims(np.concatenate([layer[i, :] for layer in params], axis=0), axis=0)])

        for n_layer in range(mlp.n_layers_ - 1):
            params_diff = np.mean(np.abs(np.diff(params[n_layer], axis=0)), axis=1)
            results[n_rep, n_layer, 0, n_v] = np.max(params_diff[noise_range[0]:noise_range[1]])

        for n_layer in range(mlp.n_layers_ - 1):
            params_diff = np.linalg.norm(np.diff(params[n_layer], axis=0), axis=1)/(np.linalg.norm(params[n_layer][:-1], axis=1))
            results[n_rep, n_layer, 1, n_v] = np.max(params_diff[noise_range[0]:noise_range[1]])

np.save("results.npy", results)
np.save("results_max_performance_loss.npy", results_max_performance_loss)