import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# random data
observations = 1000

xs = np.random.uniform(-10, 10, [observations, 1])
zs = np.random.uniform(-10, 10, [observations, 1])

inputs = np.column_stack((xs, zs))

print(inputs.shape)

# targets
noise = np.random.uniform(-1, 1, [observations, 1])
targets = 2 * xs + 3 * zs + noise
print(targets.shape)

# plot the training data
targets = targets.reshape(observations, )
xs = xs.reshape(observations, )
zs = zs.reshape(observations, )
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, zs, targets)
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('targets')
ax.view_init(azim=100)
plt.show()
targets = targets.reshape(observations, 1)

# initial variables
init_range = 0.1
weights = np.random.uniform(-init_range, init_range, [2, 1])
biases = np.random.uniform(-init_range, init_range, size=1)

print(weights)
print(biases)

# learning rate
learning_rate = 0.02

# training the model
for i in range(100):
    outputs = np.dot(inputs, weights) + biases  # X*w + b
    deltas = outputs - targets

    loss = np.sum(np.sqrt(deltas ** 2)) / observations
    print('loss', loss)

    deltas_scaled = deltas / observations
    weights = weights - learning_rate * np.dot(inputs.T, deltas_scaled)
    biases = biases - learning_rate * np.sum(deltas_scaled)

# print weight and biases to verify
print(weights, biases)

# plot last outputs and targets
plt.plot(outputs, targets)
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()
