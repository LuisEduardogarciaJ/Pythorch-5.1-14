#-----------------------------------------
#Garcia Junquera Luis Eduardo
#10
#-----------------------------------------

'''
Las transformaciones se pueden aplicar a imágenes PIL, tensores, ndarrays o datos personalizados
durante la creación del DataSet

lista completa de transformaciones integradas:
https://pytorch.org/docs/stable/torchvision/transforms.html

en imágenes
---------
CenterCrop, Escala de grises, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Redimensionar, Escalar

Sobre tensores
----------
Transformación lineal, normalización, borrado aleatorio

Conversión
----------
ToPILImage: de tensor o ndrarray
ToTensor: de numpy.ndarray o PILImage

Genérico
-------
Usar lambda

Costumbre
------
Escribir clase propia

Componer múltiples transformaciones
---------------------------
compuesta = transforma.Componer([Reescalar(256),
                               Cultivo aleatorio(224)])
'''

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np


class WineDataset(Dataset):

    def __init__(self, transform=None):
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        #--------------------------------------------------------
        # tenga en cuenta que aquí no convertimos a tensor
        #----------------------------------------------------------
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

#--------------------------------------
# Transformaciones comunes
# implementa __call__(self, sample)
#----------------------------------------
class ToTensor:
    # Convertir ndarrays to Tensores
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    # multiplica inputs con los factor
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

print('Sin transformacion')
dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nCon Tensores Transform')
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nCon Tensores y Tranformacion de la Multiplicacion ')
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)


