import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.neural_network import MLPClassifier
from generator.OWDSG import OWDSG

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

magnitude_max = 5
magnitude_step = 0.1

magnitudes = np.linspace(0, magnitude_max, int(round(magnitude_max / magnitude_step)) + 1)

n_reps = 20

n_layers = 4

results = np.zeros((n_reps, len(magnitudes), 2))
results_layers = np.zeros((n_reps, n_layers, len(magnitudes), 2))
results_webb = np.zeros((n_reps, len(magnitudes), 3, 2))
random_states = np.zeros((n_reps * len(magnitudes)))

n_chunks = 300
batch_size = 100
n_features = 100
n_informative_percent = 0.8
n_classes = 10

for n_m, magnitude in enumerate(magnitudes):
    print('n_m:', n_m)

    for n_rep in range(n_reps):
        print('n_rep:', n_rep)

        random_state = np.random.randint(0, 42949672)
        random_states[n_m * n_reps + n_rep] = random_state

        saving_range = 5
        save_1_index = 100
        save_2_index = 299
        saved_1 = [[], []]
        saved_2 = [[], []]

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

            if save_1_index - saving_range < n_chunk <= save_1_index:
                saved_1[0].append(X)
                saved_1[1].append(y)

            if save_2_index - saving_range < n_chunk <= save_2_index:
                saved_2[0].append(X)
                saved_2[1].append(y)

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

        saved_1[0] = np.concatenate(saved_1[0])
        saved_1[1] = np.concatenate(saved_1[1])

        saved_2[0] = np.concatenate(saved_2[0])
        saved_2[1] = np.concatenate(saved_2[1])

        covariate_drifts = []
        conditioned_marginal_covariate_drifts = []
        posterior_drifts = []

        for i in range(n_features):
            covariate_drifts.append(dm(saved_1[0][:, i].reshape(-1, 1), saved_2[0][:, i].reshape(-1, 1)))
            conditioned_marginal_covariate_drifts.append(cmcd(saved_1[0][:, i].reshape(-1, 1), saved_1[1], saved_2[0][:, i].reshape(-1, 1), saved_2[1]))
            posterior_drifts.append(pd(saved_1[0][:, i].reshape(-1, 1), saved_1[1], saved_2[0][:, i].reshape(-1, 1), saved_2[1]))

        results_webb[n_rep, n_m, 0, 0] = np.nanmax(covariate_drifts)
        results_webb[n_rep, n_m, 1, 0] = np.nanmax(conditioned_marginal_covariate_drifts)
        results_webb[n_rep, n_m, 2, 0] = np.nanmax(posterior_drifts)

        results_webb[n_rep, n_m, 0, 1] = np.nanmean(covariate_drifts)
        results_webb[n_rep, n_m, 1, 1] = np.nanmean(conditioned_marginal_covariate_drifts)
        results_webb[n_rep, n_m, 2, 1] = np.nanmean(posterior_drifts)

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
np.save("results_webb.npy", results_webb)