import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from generator.OWDSG import OWDSG
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'monospace'

def max_performance_loss(acc, drift, window_size=10):
    window = acc[max(0, drift - window_size):drift + window_size]

    return np.max(window) - np.min(window)

n_chunks = 200
batch_size = 200
n_features = 100
n_informative_percent = 0.8
n_classes = 6
n_layers = 4

stream = OWDSG(n_drifts=1,
                n_novel=0,
                n_classes=n_classes,
                n_features=n_features,
                n_informative=int(n_features * n_informative_percent),
                n_chunks=n_chunks,
                chunk_size=batch_size,
                random_state=6626)

mlp = MLPClassifier(hidden_layer_sizes=(40, 40, 40, ), batch_size=batch_size, random_state=553)

accuracies = []
first = True
params = []

for n_chunk_stream in range(n_chunks):
    X, y = stream.get_chunk(n_chunk_stream)

    if first:
        mlp.partial_fit(X, y, classes=np.arange(n_classes))

        for n_layer in range(mlp.n_layers_ - 1):
            params.append(np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()]))
            params[n_layer] = np.expand_dims(params[n_layer], axis=0)

        first = False
    
    else:

        acc = accuracy_score(y, mlp.predict(X))
        accuracies.append(acc)

        mlp.partial_fit(X, y, classes=np.arange(n_classes))

        for n_layer in range(mlp.n_layers_ - 1):
            params[n_layer] = np.concat([params[n_layer], np.expand_dims(np.concatenate([mlp.coefs_[n_layer].ravel(), mlp.intercepts_[n_layer].ravel()]), axis=0)])


# ---
results_params_layers = np.load("experiments/main/exp2/results_params.npy")
results_magnitudes = np.load("experiments/main/exp2/results_magnitude.npy")

print(results_params_layers.shape)
print(results_magnitudes.shape)

results_params_layers = results_params_layers[:,:,:,1].swapaxes(0,1).reshape(4,-1)
results_magnitudes = results_magnitudes[:,:,0].reshape(1,-1)

print(results_params_layers.shape) # layers x (reps x all chunks)  (relative)
print(results_magnitudes.shape)# 1 x (reps x all chunks) (max perf loss)

# -----

fig, ax = plt.subplots(1,2, figsize=(15,4), width_ratios=[3,1])

ax2 = ax[0].twinx()
ax2.plot(np.arange(1,200),accuracies, color=plt.cm.coolwarm(1.0))

mpl = max_performance_loss(accuracies, stream.get_drift_gt()[0])

params_net = None
for i in range(params[0].shape[0]):
    if params_net is None:
        params_net =  np.expand_dims(np.concatenate([layer[i, :] for layer in params], axis=0), axis=0)
    params_net = np.concat([params_net, np.expand_dims(np.concatenate([layer[i, :] for layer in params], axis=0), axis=0)])

cols = plt.cm.Blues_r([0.0,0.15,0.4,0.5])
rpcs = []
for n_layer in range(mlp.n_layers_ - 1):
    params_diff = np.linalg.norm(np.diff(params[n_layer], axis=0), axis=1)/(np.linalg.norm(params[n_layer][:-1], axis=1))
    ax[0].plot(np.arange(1,200), params_diff, 
               label='RPC layer %i' % (n_layer+1),
               color=cols[n_layer])
    
    rpcs.append(np.max(params_diff[100:150]))

ax[0].set_title('Max performance loss = %0.3f | Max relative parameter change = %0.5f (Layer %i)' % (mpl, np.max(rpcs), np.argmax(rpcs)+1))

ax2.set_ylim(-0.85, 0.7)
ax2.set_yticks(np.linspace(0,0.7,8))
ax2.set_ylabel('Accuracy', color=plt.cm.coolwarm(1.0))
ax2.yaxis.set_label_coords(x=1.05, y=0.74)
ax2.hlines(0,200,0,color='gray', lw=0.7)

ax[0].grid(ls=':')
ax[0].legend(ncols=4, frameon=False, loc='lower center')
ax[0].set_ylim(0, 0.007)
ax[0].set_ylabel('Relative parameter change', color=plt.cm.Blues(1.0))
ax[0].set_xlabel('chunk')


ax[0].set_xlim(0,200)
ax[0].spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)


# ----
for layer in range(4):
    ax[1].scatter(results_magnitudes, 
                  results_params_layers[layer], 
                  color=cols[layer],
                  s=5)
    
corr = spearmanr(results_magnitudes[0], 
                 results_params_layers[0]).statistic
print(corr)
    
ax[1].grid(ls=':')
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_title('Spearman corr. coef = %0.3f' % corr)

ax[1].set_xlabel('Max performance loss')
ax[1].set_ylabel('Relative parameter change')

plt.tight_layout()
# plt.savefig('foo.png')
plt.savefig('figures/example_corr.png')
plt.savefig('figures/example_corr.pdf')