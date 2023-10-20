#Perceptrón, ADALINE y MADALINE

import numpy as np

# Definición de la clase Perceptrón
class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        # Inicialización de hiperparámetros
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        # Inicialización de pesos y lista para almacenar errores
        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.n_iterations):
            error = 0
            for xi, target in zip(X, y):
                # Actualización de pesos basada en el error
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                error += int(update != 0.0)
            self.errors.append(error)
        return self

    def net_input(self, X):
        # Cálculo de la entrada neta
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        # Función de activación para predecir la clase
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Definición de la clase ADALINE (Adaptive Linear Neuron)
class AdalineGD:
    def __init__(self, learning_rate=0.01, n_iterations=50):
        # Inicialización de hiperparámetros
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        # Inicialización de pesos y lista para almacenar costos
        self.weights = np.zeros(1 + X.shape[1])
        self.costs = []

        for _ in range(self.n_iterations):
            # Cálculo de la entrada neta
            net_input = self.net_input(X)
            # Función de activación (en este caso, la identidad)
            output = self.activation(net_input)
            # Cálculo de errores
            errors = y - output
            # Actualización de pesos basada en el gradiente descendente
            self.weights[1:] += self.learning_rate * X.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()
            # Cálculo del costo (MSE)
            cost = (errors**2).sum() / 2.0
            self.costs.append(cost)
        return self

    def net_input(self, X):
        # Cálculo de la entrada neta
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, X):
        # Función de activación (en este caso, la identidad)
        return X

    def predict(self, X):
        # Función de predicción basada en la función de activación
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

# Definición de la clase MADALINE (Multiple ADALINE)
class Madaline:
    def __init__(self, learning_rate=0.01, n_iterations=50, n_neurons=2):
        # Inicialización de hiperparámetros y creación de múltiples ADALINEs
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_neurons = n_neurons
        self.adalines = [AdalineGD(learning_rate, n_iterations) for _ in range(n_neurons)]

    def fit(self, X, y):
        # Entrenamiento de cada ADALINE en función de los datos de entrada y las salidas deseadas
        for i in range(self.n_neurons):
            self.adalines[i].fit(X, y[i])
        return self

    def predict(self, X):
        # Predicción utilizando múltiples ADALINEs
        predictions = np.array([adaline.predict(X) for adaline in self.adalines]).T
        return predictions
