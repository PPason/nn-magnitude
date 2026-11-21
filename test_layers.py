import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from generator.OWDSG import OWDSG

n_chunks = 600
batch_size = 100
n_features = 40
n_informative_percent = 0.8
n_classes = 10
drift_magnitude = 0.5

X_noise = False
X_noise_range = [300, 310]
X_noise_factor = 2

y_noise = True
y_noise_range = [300, 310]
y_noise_factor = 0.75

stream = OWDSG(n_drifts=0,
               n_novel=0,
               n_classes=n_classes,
               n_features=n_features,
               n_informative=int(n_features * n_informative_percent),
               n_chunks=n_chunks,
               chunk_size=batch_size,
               drift_magnitude=drift_magnitude,
               smooth_drift=True)

mlp = MLPClassifier(hidden_layer_sizes=(40, 40, 40, ), batch_size=batch_size)

accuracies = []

first = True

params = None

for n_chunk in range(n_chunks):
    X, y = stream.get_chunk(n_chunk)

    if X_noise and (X_noise_range[0] <= n_chunk <= X_noise_range[1]):
        X += X_noise_factor * np.abs(X) * np.random.normal(0, 1.0, size=X.shape)

    if y_noise and (y_noise_range[0] <= n_chunk <= y_noise_range[1]):
        flip_mask = np.random.random(len(y)) < y_noise_factor
        y[flip_mask] = np.random.randint(0, n_classes, size=flip_mask.sum())

    if first:
        mlp.partial_fit(X, y, classes=np.arange(n_classes))

        params_net = None

        for n_layer in range(mlp.n_layers_ - 1):
            if params_net is None:
                params_net = np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()])
                continue

            params_net = np.concatenate([params_net, np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()])])

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

        params_net = np.concatenate([params_net, np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()])])

    params_net = np.expand_dims(params_net, axis=0)

    params = np.concat([params, params_net], axis=0)

fig, ax  = plt.subplot_mosaic([['acc', 'acc'],
                               ['1', '2']])

params_diff = np.mean(np.abs(np.diff(params, axis=0)), axis=1)

ax['1'].plot(params_diff)
ax['1'].set_title(f'Absolute mean')
ax['1'].set_xlabel('N-th chunk')

params_diff = np.linalg.norm(np.diff(params, axis=0), axis=1)/np.linalg.norm(params[:-1], axis=1)

ax['2'].plot(params_diff)
ax['2'].set_title(f'Relative norm change')
ax['2'].set_xlabel('N-th chunk')

ax['acc'].plot(accuracies)
ax['acc'].set_xlabel('N-th chunk')
ax['acc'].set_title("Accuracy")
ax['acc'].grid()

plt.show()