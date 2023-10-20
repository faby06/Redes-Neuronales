#Retropropagación del Error

import numpy as np

# Definición de la función de activación (sigmoide)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Datos de entrada y salida
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inicialización de pesos y sesgos
input_dim = 2
hidden_dim = 4
output_dim = 1

np.random.seed(1)

weights_input_hidden = np.random.uniform(-1, 1, (input_dim, hidden_dim))
bias_hidden = np.zeros((1, hidden_dim))

weights_hidden_output = np.random.uniform(-1, 1, (hidden_dim, output_dim))
bias_output = np.zeros((1, output_dim))

# Hiperparámetros
learning_rate = 0.1
epochs = 10000

# Entrenamiento
for epoch in range(epochs):
    # Propagación hacia adelante
    input_layer = X
    hidden_layer_input = np.dot(input_layer, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)

    # Cálculo de la pérdida (error cuadrático medio)
    loss = 0.5 * np.mean((output_layer_output - y) ** 2)

    # Retropropagación
    error_output = y - output_layer_output
    delta_output = error_output * sigmoid_derivative(output_layer_output)

    error_hidden = delta_output.dot(weights_hidden_output.T)
    delta_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Actualización de pesos y sesgos
    weights_hidden_output += hidden_layer_output.T.dot(delta_output) * learning_rate
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += input_layer.T.dot(delta_hidden) * learning_rate
    bias_hidden += np.sum(delta_hidden, axis=0) * learning_rate

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}: Loss {loss}')

# Predicciones
print("Predicciones:")
print(output_layer_output)

