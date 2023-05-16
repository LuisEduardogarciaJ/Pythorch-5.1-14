#--------------------------------------------
#Garcia Junquera Luis Eduardo
#9
#-------------------------------------------

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches

'''
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''

# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

# --> DataLoader can do the batch computation for us

# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__
#---------------------------------------------------------------------------------------------------------------------------------
# cálculo de gradiente, etc. no es eficiente para todo el conjunto de datos
# -> dividir el conjunto de datos en lotes pequeños

'''
# bucle de entrenamiento
para la época en el rango (num_epochs):
    # recorrer todos los lotes
    para i en el rango (total_lotes):
        lote_x, lote_y = ...
'''

# epoch = un pase hacia adelante y hacia atrás de TODAS las muestras de entrenamiento
# lote_tamaño = número de muestras de entrenamiento utilizadas en un pase de avance/retroceso
# número de iteraciones = número de pases, cada pase (adelante+atrás) usando [batch_size] número de muestras
# por ejemplo: 100 muestras, batch_size=20 -> 100/20=5 iteraciones para 1 época

# --> DataLoader puede hacer el cálculo por lotes por nosotros

# Implementar un conjunto de datos personalizado:
# heredar conjunto de datos
# implementar __init__, __getitem__ y __len__

#---------------------------------------------------------------------------------------------------------------------------------

class WineDataset(Dataset):

    def __init__(self):
       # Inicializar datos, descargar, etc.
        # leer con numpy o pandas
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

#-------------------------------------------------------------------------------------------------
        # aquí la primera columna es la etiqueta de la clase, el resto son las características
        #-----------------------------------------------------------------------------------------
        
        self.x_data = torch.from_numpy(xy[:, 1:]) # Tamaño [n_ejemplos, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]]) # Tamaño [n_ejemplos, 1]
        
# admitir la indexación de modo que el conjunto de datos [i] se pueda usar para obtener la i-ésima muestra
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    #LLamamos len(dataset) y regresa el size
    def __len__(self):
        return self.n_samples

#-----------------------
# Crear una dataset
#-----------------------
dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(features, labels)

# Cargar todo el conjunto de datos con DataLoader
# shuffle: datos aleatorios, bueno para entrenamiento
# num_workers: carga más rápida con múltiples subprocesos
# !!! SI RECIBE UN ERROR DURANTE LA CARGA, CONFIGURE num_workers EN 0 !!!
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=2)

#------------------------------------------------------
# Convertir el iterador y look en un ejemplo facil
#------------------------------------------------------
dataiter = iter(train_loader)
data = next(dataiter)
features, labels = data
print(features, labels)

#--------------------------------------
# Ciclo de entrenamiento ficticio
#--------------------------------------
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        #----------------------------------------------------------------------------
        # aquí: 178 muestras, batch_size = 4, n_iters=178/4=44.5 -> 45 iteraciones
        # Ejecuta tu proceso de entrenamiento
        #----------------------------------------------------------------------------
        if (i+1) % 5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')

#-----------------------------------------------------------------------------------
## algunos conjuntos de datos famosos están disponibles en torchvision.datasets
# p.ej. MNIST, Moda-MNIST, CIFAR10, COCO
#-----------------------------------------------------------------------------------

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),  
                                           download=True)

train_loader = DataLoader(dataset=train_dataset, 
                                           batch_size=3, 
                                           shuffle=True)

#--------------------------------------
# Saca una muestra aleatoria
#--------------------------------------
dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)



