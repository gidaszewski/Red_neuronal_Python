import numpy as np
import math
import matplotlib.pyplot as plt

# /// Función ReLu
# La función ReLu es muy simple: para valores negativos, la función devuelve cero.
# Para valores positivos, la función devuelve el mismo valor. Pero, a pesar de ser tan simple,
# esta función es la función de activación más usada en el campo de las redes neuronales y deep learning.

def derivada_relu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

relu = (
    lambda x: x * ( x > 0),
    lambda x: derivada_relu(x)
)

rango = np.linspace(-10, 10).reshape([50,1])
datos_relu = relu[0](rango)
datos_relu_derivada = relu[1](rango)

rango = np.linspace(-10, 10).reshape([50,1])

"""# Gráficos
plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (15,5))
axes[0].plot(rango, datos_relu[:,0])
axes[1].plot(rango, datos_relu_derivada[:,0])
plt.show()"""