#Perceptr�n, ADALINE y MADALINE

import numpy as np

# Definici�n de la clase Perceptr�n
class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        # Inicializaci�n de hiperpar�metros
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        # Inicializaci�n de pesos y lista para almacenar errores
        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.n_iterations):
            error = 0
            for xi, target in zip(X, y):
                # Actualizaci�n de pesos basada en el error
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                error += int(update != 0.0)
            self.errors.append(error)
        return self

    def net_input(self, X):
        # C�lculo de la entrada neta
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        # Funci�n de activaci�n para predecir la clase
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Definici�n de la clase ADALINE (Adaptive Linear Neuron)
class AdalineGD:
    def __init__(self, learning_rate=0.01, n_iterations=50):
        # Inicializaci�n de hiperpar�metros
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        # Inicializaci�n de pesos y lista para almacenar costos
        self.weights = np.zeros(1 + X.shape[1])
        self.costs = []

        for _ in range(self.n_iterations):
            # C�lculo de la entrada neta
            net_input = self.net_input(X)
            # Funci�n de activaci�n (en este caso, la identidad)
            output = self.activation(net_input)
            # C�lculo de errores
            errors = y - output
            # Actualizaci�n de pesos basada en el gradiente descendente
            self.weights[1:] += self.learning_rate * X.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()
            # C�lculo del costo (MSE)
            cost = (errors**2).sum() / 2.0
            self.costs.append(cost)
        return self

    def net_input(self, X):
        # C�lculo de la entrada neta
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, X):
        # Funci�n de activaci�n (en este caso, la identidad)
        return X

    def predict(self, X):
        # Funci�n de predicci�n basada en la funci�n de activaci�n
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

# Definici�n de la clase MADALINE (Multiple ADALINE)
class Madaline:
    def __init__(self, learning_rate=0.01, n_iterations=50, n_neurons=2):
        # Inicializaci�n de hiperpar�metros y creaci�n de m�ltiples ADALINEs
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_neurons = n_neurons
        self.adalines = [AdalineGD(learning_rate, n_iterations) for _ in range(n_neurons)]

    def fit(self, X, y):
        # Entrenamiento de cada ADALINE en funci�n de los datos de entrada y las salidas deseadas
        for i in range(self.n_neurons):
            self.adalines[i].fit(X, y[i])
        return self

    def predict(self, X):
        # Predicci�n utilizando m�ltiples ADALINEs
        predictions = np.array([adaline.predict(X) for adaline in self.adalines]).T
        return predictions
