import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from generator.OWDSG import OWDSG

n_chunks = 100
batch_size = 100
n_features = 100
n_informative_percent = 0.8
n_classes = 10

X_noise = False
X_noise_range = [200, 230]
X_noise_factor = 0.5

y_noise = False
y_noise_range = [125, 155]
y_noise_factor = 1

stream = OWDSG(n_drifts=1,
               n_novel=0,
               hide_label=True,
               n_classes=n_classes,
               n_features=n_features,
               n_informative=int(n_features * n_informative_percent),
               n_chunks=n_chunks,
               chunk_size=batch_size,
               drift_magnitude=0.1,
               smooth_drift=False)

mlp = MLPClassifier(hidden_layer_sizes=(40, 40, 40,), batch_size=batch_size)

accuracies = []

first = True

params = []

for n_chunk in range(n_chunks):
    X, y = stream.get_chunk(n_chunk)

    if X_noise and (X_noise_range[0] <= n_chunk <= X_noise_range[1]):
        X += X_noise_factor * np.abs(X) * np.random.normal(0, 1.0, size=X.shape)

    if y_noise and (y_noise_range[0] <= n_chunk <= y_noise_range[1]):
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

params_net = None

for i in range(params[0].shape[0]):
    if params_net is None:
        params_net =  np.expand_dims(np.concatenate([layer[i, :] for layer in params], axis=0), axis=0)
        continue

    params_net = np.concat([params_net, np.expand_dims(np.concatenate([layer[i, :] for layer in params], axis=0), axis=0)])

fig, ax  = plt.subplot_mosaic([['acc', 'acc'],
                               ['1_layer', '2_layer'],
                               ['1_net', '2_net']])

for n_layer in range(mlp.n_layers_ - 1):
    params_diff = np.mean(np.abs(np.diff(params[n_layer], axis=0)), axis=1)

    ax['1_layer'].plot(params_diff, label=f'Layer {n_layer}')
    ax['1_layer'].set_title(f'Absolute mean')
    ax['1_layer'].set_xlabel('N-th chunk')

ax['1_layer'].legend()

for n_layer in range(mlp.n_layers_ - 1):
    params_diff = np.linalg.norm(np.diff(params[n_layer], axis=0), axis=1)/(np.linalg.norm(params[n_layer][:-1], axis=1))

    ax['2_layer'].plot(params_diff, label=f'Layer {n_layer}')
    ax['2_layer'].set_title(f'Relative norm change')
    ax['2_layer'].set_xlabel('N-th chunk')

ax['2_layer'].legend()

params_diff = np.mean(np.abs(np.diff(params_net, axis=0)), axis=1)

ax['1_net'].plot(params_diff)
ax['1_net'].set_title(f'Absolute mean')
ax['1_net'].set_xlabel('N-th chunk')

params_diff = np.linalg.norm(np.diff(params_net, axis=0), axis=1)/np.linalg.norm(params_net[:-1], axis=1)

ax['2_net'].plot(params_diff)
ax['2_net'].set_title(f'Relative norm change')
ax['2_net'].set_xlabel('N-th chunk')

ax['acc'].plot(accuracies)
ax['acc'].set_xlabel('N-th chunk')
ax['acc'].set_title("Accuracy")
ax['acc'].grid()

plt.show()