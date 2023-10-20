#Mapas Autoorganizados de Kohonen

import numpy as np
import matplotlib.pyplot as plt

# Función para inicializar los pesos del SOM
def initialize_som(input_dim, map_size):
    return np.random.rand(map_size[0], map_size[1], input_dim)

# Función para encontrar la neurona ganadora (BMU - Best Matching Unit)
def find_bmu(data_point, som_weights):
    # Calcular las distancias entre el dato de entrada y todos los pesos de las neuronas
    distances = np.linalg.norm(som_weights - data_point, axis=2)
    # Encontrar las coordenadas de la BMU (la neurona con la distancia más pequeña)
    bmu_coords = np.unravel_index(np.argmin(distances), distances.shape)
    return bmu_coords

# Función para actualizar los pesos del SOM
def update_som_weights(som_weights, data_point, bmu_coords, learning_rate, radius):
    for i in range(som_weights.shape[0]):
        for j in range(som_weights.shape[1]):
            weight = som_weights[i, j]
            distance_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_coords))
            if distance_to_bmu <= radius:
                influence = np.exp(-(distance_to_bmu ** 2) / (2 * (radius ** 2)))
                weight += learning_rate * influence * (data_point - weight)
            som_weights[i, j] = weight
    return som_weights

# Generar datos de ejemplo
data = np.array([[0.1, 0.2, 0.3],
                 [0.6, 0.7, 0.8],
                 [0.9, 0.45, 0.2],
                 [0.8, 0.5, 0.1]])

# Hiperparámetros
input_dim = data.shape[1]
map_size = (5, 5)  # Tamaño del mapa de neuronas
epochs = 100
learning_rate = 0.1
initial_radius = max(map_size) / 2

# Inicializar pesos del SOM
som_weights = initialize_som(input_dim, map_size)

# Entrenamiento del SOM
for epoch in range(epochs):
    for data_point in data:
        bmu_coords = find_bmu(data_point, som_weights)
        som_weights = update_som_weights(som_weights, data_point, bmu_coords, learning_rate, initial_radius * (1 - epoch / epochs))

# Resultados del SOM
print("Mapa de neuronas (pesos):")
print(som_weights)

# Visualización del SOM
plt.imshow(som_weights, interpolation='none')
plt.show()

