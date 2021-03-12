import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Data Generation

observation = 1000
xs = np.random.uniform(-10, 10, [observation, 1])
zs = np.random.uniform(-10, 10, [observation, 1])

generated_inputs = np.column_stack((xs, zs))

noise = np.random.uniform(-1, 1, [observation, 1])

generated_targets = 2 * xs - 3 * zs + 5 + noise

np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)

# Solving with TensorFlow

training_data = np.load('TF_intro.npz')

input_size = 2
output_size = 1

model = tf.keras.Sequential([
    tf.keras.layers.Dense(output_size,
                          kernel_initializer=tf.random_uniform_initializer(minval=-0.1),
                          bias_initializer=tf.random_uniform_initializer(minval=-0.1))

])

custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=1)

# Extract the weights and bias

model.layers[0].get_weights()

weights = model.layers[0].get_weights()[0]
print(weights)

bias = model.layers[0].get_weights()[1]
print(bias)

# plotting the data

plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()
