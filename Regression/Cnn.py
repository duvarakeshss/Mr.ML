
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class CNN:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.model = self.create_model()
        self.loss_history = []

    def create_model(self):
        model = tf.keras.Sequential()
        # Conv1D layer with 2 input features and 1 channel
        model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=2, activation='relu', input_shape=(2, 1)))
        
        # Removed MaxPooling1D due to small input size
        # model.add(tf.keras.layers.MaxPooling1D(pool_size=2))  # No need for pooling in this case

        # Flatten the result and add a Dense layer
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))
        
        # Compile model with the correct learning rate
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model

    def train(self, X, y, epochs=10, batch_size=16):
        print(f"Features shape before training: {X.shape}")
        print(f"Labels shape before training: {y.shape}")

        # Fit the model
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        self.loss_history = history.history['loss']

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, label='Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xticks(np.arange(1, len(self.loss_history) + 1, 1))
        plt.legend()
        plt.grid()
        plt.show()




# # Convolution Layer
# class Conv1D:
#     def __init__(self, num_filters, filter_size):
#         self.num_filters = num_filters
#         self.filter_size = filter_size
#         # Initialize filters with random values
#         self.filters = np.random.randn(num_filters, filter_size) / filter_size

#     def iterate_regions(self, input):
#         # Generates all possible regions using the filter size
#         for i in range(input.shape[0] - self.filter_size + 1):
#             region = input[i:i+self.filter_size]
#             yield region, i

#     def forward(self, input):
#         # Output feature map
#         self.last_input = input
#         output = np.zeros((input.shape[0] - self.filter_size + 1, self.num_filters))

#         for region, i in self.iterate_regions(input):
#             output[i] = np.sum(region * self.filters, axis=1)

#         return output

#     def backward(self, dL_dout, learning_rate):
#         # Backpropagation through convolution
#         dL_dfilters = np.zeros(self.filters.shape)

#         for region, i in self.iterate_regions(self.last_input):
#             for f in range(self.num_filters):
#                 dL_dfilters[f] += dL_dout[i, f] * region

#         self.filters -= learning_rate * dL_dfilters




# # Max Pooling Layer
# class MaxPool1D:
#     def __init__(self, pool_size):
#         self.pool_size = pool_size

#     def iterate_regions(self, input):
#         for i in range(0, input.shape[0], self.pool_size):
#             region = input[i:i+self.pool_size]
#             yield region, i

#     def forward(self, input):
#         self.last_input = input
#         output = np.zeros((input.shape[0] // self.pool_size, input.shape[1]))

#         for region, i in self.iterate_regions(input):
#             output[i // self.pool_size] = np.amax(region, axis=0)

#         return output

#     def backward(self, dL_dout):
#         # Backpropagation for Max Pooling Layer
#         dL_dinput = np.zeros(self.last_input.shape)

#         for region, i in self.iterate_regions(self.last_input):
#             h, w = region.shape
#             amax = np.amax(region, axis=0)

#             for j in range(h):
#                 for k in range(w):
#                     if region[j, k] == amax[k]:
#                         dL_dinput[i + j, k] = dL_dout[i // self.pool_size, k]

#         return dL_dinput




# # Fully Connected Layer (Dense)
# class Dense:
#     def __init__(self, input_len, output_len):
#         # Initialize weights and biases
#         self.weights = np.random.randn(input_len, output_len) / input_len
#         self.biases = np.zeros(output_len)

#     def forward(self, input):
#         self.last_input_shape = input.shape
#         self.last_input = input.flatten()
#         self.last_output = np.dot(self.last_input, self.weights) + self.biases
#         return self.last_output

#     def backward(self, dL_dout, learning_rate):
#         # Backpropagation through fully connected layer
#         dL_dinput = np.dot(dL_dout, self.weights.T)
#         dL_dW = np.dot(self.last_input[np.newaxis].T, dL_dout[np.newaxis])
#         dL_db = dL_dout

#         self.weights -= learning_rate * dL_dW
#         self.biases -= learning_rate * dL_db

#         return dL_dinput.reshape(self.last_input_shape)


# # ReLU Activation Function
# class ReLU:
#     def forward(self, input):
#         self.last_input = input
#         return np.maximum(0, input)

#     def backward(self, dL_dout):
#         dL_dinput = dL_dout.copy()
#         dL_dinput[self.last_input <= 0] = 0
#         return dL_dinput


# # Mean Squared Error Loss
# class MSE:
#     def forward(self, predicted, actual):
#         return np.mean((predicted - actual) ** 2)

#     def backward(self, predicted, actual):
#         return 2 * (predicted - actual) / actual.size

# class CNN:
#     def __init__(self):
#         self.conv1 = Conv1D(num_filters=4, filter_size=1)  # Use a filter size of 1
#         self.pool1 = MaxPool1D(pool_size=1)  # Adjust pooling to fit the input
#         self.relu1 = ReLU()
#         self.fc1 = None  # Dynamically created later
#         self.mse = MSE()
#         self.loss_history = []

#     def forward(self, input):
#         out = self.conv1.forward(input)
#         out = self.pool1.forward(out)
#         out = self.relu1.forward(out)

#         if self.fc1 is None:
#             input_len = out.shape[0] * out.shape[1]  # Dynamic calculation for Dense input
#             self.fc1 = Dense(input_len=input_len, output_len=1)

#         out = self.fc1.forward(out)
#         return out

#     def train(self, X, y, epochs=10, learning_rate=0.001):
#         for epoch in range(epochs):
#             loss = 0
#             for i in range(len(X)):
#                 output = self.forward(X[i])
#                 loss += self.mse.forward(output, y[i])
#                 grad = self.mse.backward(output, y[i])
#                 grad = self.fc1.backward(grad, learning_rate)
#                 grad = self.relu1.backward(grad)
#                 grad = self.pool1.backward(grad)
#                 self.conv1.backward(grad, learning_rate)

#             self.loss_history.append(loss / len(X))
#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss / len(X)}")

#     def predict(self, X):
#         return np.array([self.forward(x) for x in X])
