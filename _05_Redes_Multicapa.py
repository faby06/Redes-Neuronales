#Redes Multicapa

import numpy as np

# Definición de la función de activación (ReLU)
def relu(x):
    return np.maximum(0, x)

# Definición de la derivada de la función de activación ReLU
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Datos de entrada y salida
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inicialización de pesos y sesgos para dos capas (una capa oculta)
input_dim = 2
hidden_dim = 4
output_dim = 1

# Pesos y sesgos de la capa oculta
weights_hidden = np.random.rand(input_dim, hidden_dim)
bias_hidden = np.zeros((1, hidden_dim))

# Pesos y sesgos de la capa de salida
weights_output = np.random.rand(hidden_dim, output_dim)
bias_output = np.zeros((1, output_dim))

# Hiperparámetros
learning_rate = 0.1
epochs = 10000

# Entrenamiento
for epoch in range(epochs):
    # Propagación hacia adelante
    hidden_layer_input = np.dot(X, weights_hidden) + bias_hidden
    hidden_layer_output = relu(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_output) + bias_output
    predicted_output = output_layer_input

    # Cálculo de la pérdida (error cuadrático medio)
    loss = 0.5 * np.mean((predicted_output - y) ** 2)

    # Retropropagación (gradiente descendente)
    error = predicted_output - y
    delta_output = error
    delta_hidden = delta_output.dot(weights_output.T) * relu_derivative(hidden_layer_input)

    weights_output -= hidden_layer_output.T.dot(delta_output) * learning_rate
    bias_output -= np.sum(delta_output, axis=0, keepdims=True) * learning_rate
    weights_hidden -= X.T.dot(delta_hidden) * learning_rate
    bias_hidden -= np.sum(delta_hidden, axis=0) * learning_rate

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}: Loss {loss}')

# Predicciones
print("Predicciones:")
print(predicted_output)
