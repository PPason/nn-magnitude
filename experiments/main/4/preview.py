import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier

train = pd.read_csv('KDDTrain+.txt', sep=",", header=None)
test = pd.read_csv('KDDTest+.txt', sep=",", header=None)

# Shuffle
train = train.sample(frac=1)
test = test.sample(frac=1)

X_train = train.drop(train.columns[-1], axis=1)
X_test = test.drop(test.columns[-1], axis=1)

y_train = train.iloc[:, 41]
y_test = test.iloc[:, 41]

y_train = y_train.apply(lambda x: 0 if x == 'normal' else 1)
y_test = y_test.apply(lambda x: 0 if x == 'normal' else 1)

X_train = X_train.drop(X_train.columns[-1], axis=1)
X_test = X_test.drop(X_test.columns[-1], axis=1)

X = pd.concat([X_train, X_test], axis=0, sort=False)
y = pd.concat([y_train, y_test], axis=0, sort=False)

X = pd.get_dummies(X, dtype=int)

X = X.to_numpy()
y = y.to_numpy()

chunk_size = 100
batch_size = chunk_size

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10,), batch_size=batch_size)

accuracies = []

first = True

first_X_mean = None
first_X_std = None

params = []

for n_chunk in range(int(X.shape[0]/chunk_size)):
    X_batch = X[n_chunk*chunk_size:(n_chunk+1)*chunk_size, :]
    y_batch = y[n_chunk*chunk_size:(n_chunk+1)*chunk_size]

    if first:
        first_X_mean = np.mean(X_batch, axis=0)
        first_X_std = np.std(X_batch, axis=0)
        first_X_std[first_X_std == 0] = 1

        X_batch = (X_batch - first_X_mean) / first_X_std

        mlp.partial_fit(X_batch, y_batch, classes=[0, 1])

        for n_layer in range(mlp.n_layers_ - 1):
            params.append(np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()]))
            params[n_layer] = np.expand_dims(params[n_layer], axis=0)

        first = False

        continue

    X_batch = (X_batch - first_X_mean)/first_X_std

    predictions = mlp.predict(X_batch)
    accuracies.append(balanced_accuracy_score(y_batch, predictions))

    mlp.partial_fit(X_batch, y_batch, classes=[0, 1])

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