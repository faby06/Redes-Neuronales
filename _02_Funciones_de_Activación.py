#Funciones de Activaci�n

import numpy as np

# Funci�n de activaci�n Sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Funci�n de activaci�n ReLU (Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)

# Funci�n de activaci�n Tangente Hiperb�lica (Tanh)
def tanh(x):
    return np.tanh(x)

# Funci�n de activaci�n Softmax (para capa de salida en problemas de clasificaci�n multiclase)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Ejemplos de uso de las funciones de activaci�n
x = np.array([-1.0, 0.0, 1.0, 2.0])

# Aplicar la funci�n Sigmoide
result_sigmoid = sigmoid(x)
print("Resultado de Sigmoide:")
print(result_sigmoid)

# Aplicar la funci�n ReLU
result_relu = relu(x)
print("\nResultado de ReLU:")
print(result_relu)

# Aplicar la funci�n Tangente Hiperb�lica (Tanh)
result_tanh = tanh(x)
print("\nResultado de Tangente Hiperbolica:")
print(result_tanh)

# Aplicar la funci�n Softmax
x_softmax = np.array([2.0, 1.0, 0.1])
result_softmax = softmax(x_softmax)
print("\nResultado de Softmax:")
print(result_softmax)
