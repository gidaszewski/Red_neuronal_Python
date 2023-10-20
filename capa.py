from scipy import stats
import numpy as np
from func_relu import *
from func_sigmoide import *

class Capa():
    def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
        
        self.funcion_act = funcion_act
        self.b = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale = 1, size = n_neuronas).reshape(1, n_neuronas), 3)
        self.W  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale = 1, size = n_neuronas * n_neuronas_capa_anterior).reshape(n_neuronas_capa_anterior, n_neuronas), 3)


# Numero de neuronas en cada capa
# El primer valor es el numero de columnas de la capa de entrada

neuronas = [2,4,8,1]

# Funciones de activación usadas en cada capa
funciones_activacion = [relu, relu, sigmoid]

red_neuronal = []

for paso in range(len(neuronas)-1):
    x = Capa(neuronas[paso], neuronas[paso+1], funciones_activacion[paso])
    red_neuronal.append(x)
