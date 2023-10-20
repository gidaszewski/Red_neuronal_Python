import numpy as np
import math
import matplotlib.pyplot as plt

# /// Función Sigmoide
# La función sigmoide básicamente recibe un valor x y devuelve un valor entre 0 y 1.
# Esto hace que sea una función muy interesante, ya que indica la probabilidad de un estado.
# Por ejemplo, si usamos la función sigmoide en la última capa para un problema de clasificación entre dos clases,
# la función devolverá la probabilidad de pertenencia a un grupo. In [2]:

sigmoid = (
    lambda x:1 / (1 + np.exp(-x)),
    lambda x:x * (1 - x)
)

rango = np.linspace(-10, 10).reshape([50,1])
datos_sigmoide = sigmoid[0](rango)
datos_sigmoide_derivada = sigmoid[1](rango)

"""# Gráficos
fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (15,5))
axes[0].plot(rango, datos_sigmoide)
axes[1].plot(rango, datos_sigmoide_derivada)
fig.tight_layout()"""

