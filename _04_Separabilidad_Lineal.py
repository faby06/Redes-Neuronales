#Separabilidad Lineal

import numpy as np
import matplotlib.pyplot as plt

# Generar datos linealmente separables
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.concatenate([-np.ones(50), np.ones(50)])

# Separar datos en dos clases
class1 = X[y == -1]
class2 = X[y == 1]

# Trazar datos
plt.scatter(class1[:, 0], class1[:, 1], label='Clase -1', marker='o')
plt.scatter(class2[:, 0], class2[:, 1], label='Clase 1', marker='x')

# Trazar hiperplano (en este caso, una línea recta)
x_line = np.linspace(-3, 3, 100)
y_line = -x_line  # Ecuación del hiperplano
plt.plot(x_line, y_line, 'k--', label='Hiperplano')

# Etiquetas y leyenda
plt.xlabel('Caracteristica 1')
plt.ylabel('Caracteristica 2')
plt.legend(loc='best')
plt.title('Datos Linealmente Separables')

# Mostrar el gráfico
plt.show()
