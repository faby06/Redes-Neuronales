#Funciones de Activación

import numpy as np

# Función de activación Sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Función de activación ReLU (Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)

# Función de activación Tangente Hiperbólica (Tanh)
def tanh(x):
    return np.tanh(x)

# Función de activación Softmax (para capa de salida en problemas de clasificación multiclase)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Ejemplos de uso de las funciones de activación
x = np.array([-1.0, 0.0, 1.0, 2.0])

# Aplicar la función Sigmoide
result_sigmoid = sigmoid(x)
print("Resultado de Sigmoide:")
print(result_sigmoid)

# Aplicar la función ReLU
result_relu = relu(x)
print("\nResultado de ReLU:")
print(result_relu)

# Aplicar la función Tangente Hiperbólica (Tanh)
result_tanh = tanh(x)
print("\nResultado de Tangente Hiperbolica:")
print(result_tanh)

# Aplicar la función Softmax
x_softmax = np.array([2.0, 1.0, 0.1])
result_softmax = softmax(x_softmax)
print("\nResultado de Softmax:")
print(result_softmax)
