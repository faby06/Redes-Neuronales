import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Cargar el conjunto de datos MNIST
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizar los valores de píxeles al rango [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Construir el modelo de red neuronal
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar la red neuronal
model.fit(train_images, train_labels, epochs=5)

# Evaluar el modelo
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Precisión en el conjunto de prueba: {test_accuracy}')