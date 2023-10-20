import numpy as np
from capa import *
from func_relu import *
from func_sigmoide import *
from func_coste import *
from random import shuffle

# Para que nuestra red neuronal prediga lo único que tenemos que hacer es definir los cáculos que tiene que seguir.
# Son 3 los cálculos a seguir: multiplicar los valores de entrada por la matriz de pesos W y
# sumar el parámetro bias (b) y aplicar la función de activación.

# Para multiplicar los valores de entrada por la matriz de pesos tenemos que hacer una multiplicación matricial.

X = np.round(np.random.randn(20,2),3)

""" z = X @ red_neuronal[0].W

# Sumamos el parametro bias (b)

z = z + red_neuronal[0].b

a = red_neuronal[0].funcion_act[0](z) """

# Esto lo podemos definir de forma iterativa dentro de un bucle.

output = [X]

for num_capa in range(len(red_neuronal)):
    z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b
    a = red_neuronal[num_capa].funcion_act[0](z)
    output.append(a)

Y = [0] * 10 + [1] * 10
shuffle(Y)
Y = np.array(Y).reshape(len(Y),1)

""" Backpropagation: calculando el error en cada capa """

# Definimos el learning rate
lr = 0.05

# Creamos el indice inverso para ir de derecha a izquierda
back = list(range(len(output)-1))
back.reverse()

# Creamos el vector delta donde meteremos los errores en cada capa
delta = []

for Capa in back:
  # Backprop #

  # Guardamos los resultados de la ultima capa antes de usar backprop para poder usarlas en gradient descent
  a = output[Capa+1][1]

  # Backprop en la ultima Capa 
  if Capa == back[0]:
    x = mse(a,Y)[1] * red_neuronal[Capa].funcion_act[1](a)
    delta.append(x)

  # Backprop en el resto de Capas 
  else:
    x = delta[-1] @ W_temp * red_neuronal[Capa].funcion_act[1](a)
    delta.append(x)

  # Guardamos los valores de W para poder usarlos en la iteracion siguiente
  W_temp = red_neuronal[Capa].W.transpose()

  # Gradient Descent #

  # Ajustamos los valores de los parametros de la capa
  red_neuronal[Capa].b = red_neuronal[Capa].b - delta[-1].mean() * lr
  red_neuronal[Capa].W = red_neuronal[Capa].W - (output[Capa].T @ delta[-1]) * lr


print('MSE: ' + str(mse(output[-1],Y)[0]) )
print('Estimacion: ' + str((output[-1])) )
