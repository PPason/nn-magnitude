import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

n_chunks = 300
batch_size = 100
n_features = 40
n_informative_percent = 0.8
n_classes = 10

X, y = make_classification(n_samples=1000, n_features=n_features, n_informative=int(n_features * n_informative_percent), n_classes=n_classes)

mlp = MLPClassifier(hidden_layer_sizes=(40, 40, 40, ), batch_size=batch_size)

accuracies = []

first = True

params = []

n_epochs = 300

for n_epoch in range(n_epochs):
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
    ax['1_layer'].set_xlabel('N-th epoch')

ax['1_layer'].legend()

for n_layer in range(mlp.n_layers_ - 1):
    params_diff = np.linalg.norm(np.diff(params[n_layer], axis=0), axis=1)/np.linalg.norm(params[n_layer][:-1], axis=1)

    ax['2_layer'].plot(params_diff, label=f'Layer {n_layer}')
    ax['2_layer'].set_title(f'Relative norm change')
    ax['2_layer'].set_xlabel('N-th epoch')

ax['2_layer'].legend()

params_diff = np.mean(np.abs(np.diff(params_net, axis=0)), axis=1)

ax['1_net'].plot(params_diff)
ax['1_net'].set_title(f'Absolute mean')
ax['1_net'].set_xlabel('N-th epoch')

params_diff = np.linalg.norm(np.diff(params_net, axis=0), axis=1)/np.linalg.norm(params_net[:-1], axis=1)

ax['2_net'].plot(params_diff)
ax['2_net'].set_title(f'Relative norm change')
ax['2_net'].set_xlabel('N-th epoch')

ax['acc'].plot(accuracies)
ax['acc'].set_xlabel('N-th epoch')
ax['acc'].set_title("Accuracy")
ax['acc'].grid()

plt.show()