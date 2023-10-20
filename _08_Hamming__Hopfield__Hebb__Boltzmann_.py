#Hamming, Hopfield, Hebb, Boltzmann,

import numpy as np

class HopfieldNetwork:
    def __init__(self, n):
        self.weights = np.zeros((n, n))

    def train(self, patterns):
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                if i != j:
                    for pattern in patterns:
                        self.weights[i, j] += pattern[i] * pattern[j]

    def recall(self, pattern):
        for _ in range(10):  # Número máximo de iteraciones
            for i in range(pattern.shape[0]):
                s = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if s >= 0 else -1
        return pattern

def corrupt_pattern(pattern, noise_level=0.1):
    corrupted_pattern = pattern.copy()
    num_corruptions = int(noise_level * pattern.shape[0])
    indices_to_corrupt = np.random.choice(pattern.shape[0], num_corruptions, replace=False)
    corrupted_pattern[indices_to_corrupt] = -corrupted_pattern[indices_to_corrupt]
    return corrupted_pattern

# Datos de ejemplo
patterns = np.array([[1, 1, -1, -1], [-1, -1, 1, 1], [1, -1, 1, -1]])

# Crear y entrenar la red de Hopfield
hopfield_net = HopfieldNetwork(patterns.shape[1])
hopfield_net.train(patterns)

# Recuperar patrones
for pattern in patterns:
    corrupted_pattern = corrupt_pattern(pattern)
    retrieved_pattern = hopfield_net.recall(corrupted_pattern)
    print("Patron original:", pattern)
    print("Patron corrupto:", corrupted_pattern)
    print("Patron recuperado:", retrieved_pattern)
    print()

