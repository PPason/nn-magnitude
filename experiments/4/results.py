import numpy as np

# 0 = mean abs
# 1 = norm

metric = 1

n_layers = 4

y_noise = np.arange(0, 1, 0.05)

results_max_performance_loss = np.load("results_max_performance_loss.npy")
results_params = np.load("results_params.npy")

results_max_performance_loss_mean = np.mean(results_max_performance_loss, axis=0)

print(results_max_performance_loss_mean)

baseline = np.mean(results_params[:, :, 0, :], axis=0)

how_much_more = []

for i, magnitude in enumerate(y_noise[1:]):
    how_much_more.append(np.expand_dims(results_params[:, :, i, :]/baseline, axis=2))

how_much_more = np.concat(how_much_more, axis=2)
how_much_more_mean = np.mean(how_much_more, axis=0)
how_much_more_std = np.std(how_much_more, axis=0)

print("RELATIVE")

for i, magnitude in enumerate(y_noise[1:]):
    print(f"y noise: {magnitude}, magnitude: {results_max_performance_loss_mean[i+1]}")

    for n_layer in range(n_layers):
        print(f"Layer {n_layer} | {how_much_more_mean[n_layer, i, metric]:.3f} ({how_much_more_std[n_layer, i, metric]:.3f})")

# print("\nABSOLUTE")
#
# results_params_mean = np.mean(results_params, axis=0)
# results_params_std = np.std(results_params, axis=0)
#
# for i, magnitude in enumerate(y_noise[1:]):
#     print(f"y noise: {magnitude}, magnitude: {results_max_performance_loss_mean[i+1]}")
#
#     for n_layer in range(n_layers):
#         print(f"Layer {n_layer} | {results_params_mean[n_layer, i, metric]:.3f} ({results_params_std [n_layer, i, metric]:.3f})")