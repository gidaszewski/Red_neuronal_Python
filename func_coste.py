import numpy as np

# Calcular el error cuadrático medio es algo bastante simple: a cada valor predicho le restas
# el valor real, lo elevas al cuadrado, haces la suma ponderada y calculas su raíz.
# Además, como hemos hecho anteriormente aprovecharemos para que esta misma función nos devuelva
# la derivada dela función de coste, la cual nos será útil en el paso de backpropagation.

def mse(Ypredich, Yreal):

    """Calculamos el error"""

    x = (np.array(Ypredich) - np.array(Yreal)) ** 2
    x = np.mean(x)

    """Calculamos la derivada de la funcion"""

    y = np.array(Ypredich) - np.array(Yreal)
    return(x,y)