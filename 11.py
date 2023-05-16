#---------------------------------------------------------------
#Garcia Junquera Luis Eduardo
#11
#Introduccion al uso de softmax y crss entropy loss en python
#---------------------------------------------------------------


import torch
import torch.nn as nn
import numpy as np

#--------------------------------------------------------------------------
#
#        -> 2.0              -> 0.65  
# Lineal -> 1.0 -> Softmax -> 0.25 -> CrossEntropy(y, y_hat)
#        -> 0.1              -> 0.1                   
#
# puntuaciones (logits) probabilidades
# suma = 1.0
#

# Softmax aplica la función exponencial a cada elemento, y normaliza
# dividiendo por la suma de todos estos exponenciales
# -> aplasta la salida para que esté entre 0 y 1 = probabilidad
# la suma de todas las probabilidades es 1
#---------------------------------------------------------------------------
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
#--------------------------
#Vector en R3
#-----------------------------
x = np.array([2.0, 1.0, 0.1])
#-----------------------------------
#Softmax de elemtos de vector
#----------------------------------
outputs = softmax(x)

print('softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0) # tomar softmax de los elemtos del eje 0
print('softmax torch:', outputs)

#-------------------------------------------------------------------------------------------------------------------
# Entropía cruzada
# La pérdida de entropía cruzada, o pérdida logarítmica, mide el rendimiento de un modelo de clasificación 
# cuya salida es un valor de probabilidad entre 0 y 1. 
# -> pérdida aumenta a medida que la probabilidad prevista diverge de la etiqueta real
#se incrementa conforme la probabilidad diverge del nivel verdaedero
#-------------------------------------------------------------------------------------------------------------------
def cross_entropy(actual, predicted):
    #Limita los valores a un minimo EPS
    EPS = 1e-15
    predicted = np.clip(predicted, EPS, 1 - EPS)
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])

#----------------------------------------------
# y debe ser un codificado en caliente
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
#-----------------------------------------------
Y = np.array([1, 0, 0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

#-----------------------------------------------------------
# CrossEntropyLoss en PyTorch (aplica Softmax)
# nn. LogSoftmax + nn. NLLLoss
# NLLLoss = pérdida de probabilidad logarítmica negativa
#-----------------------------------------------------------
loss = nn.CrossEntropyLoss()

#-----------------------------------------------------------
# pérdida (entrada, destino)
# el objetivo es de tamaño nSamples = 1
# Cada elemento tiene una etiqueta de clase: 0, 1 o 2
# Y (=destino) contiene etiquetas de clase, no una
#-----------------------------------------------------------
Y = torch.tensor([0])
#---------------------------------------------------------------------------------------------------------------]
# la entrada es de tamaño nSamples x nClasses = 1 x 3
# y_pred (=entrada) debe ser raw, desnormaliza las puntuaciones (logits) para cada clase, no softmax
#---------------------------------------------------------------------------------------------------------------]
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'PyTorch Loss1: {l1.item():.4f}')
print(f'PyTorch Loss2: {l2.item():.4f}')

#--------------------
# predicciones ( regresa el maximo de la 1ra dimension)
#-------------------
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y.item()}, Y_pred1: {predictions1.item()}, Y_pred2: {predictions2.item()}')
#--------------------------------------------------------------
# Permite la pérdida de lotes para múltiples muestras

# el destino es de tamaño nBatch = 3
# Cada elemento tiene una etiqueta de clase: 0, 1 o 2
Y = torch.tensor([2, 0, 1])

# la entrada es de tamaño nBatch x nClasses = 3 x 3
# Y_pred son logits (no softmax)
#-------------------------------------------------------------------
Y_pred_good = torch.tensor(
    [[0.1, 0.2, 3.9], # prediccion de clase 2
    [1.2, 0.1, 0.3], # prediccion de clase 0
    [0.3, 2.2, 0.2]]) # prediccion de clase 1

Y_pred_bad = torch.tensor(
    [[0.9, 0.2, 0.1],
    [0.1, 0.3, 1.5],
    [1.2, 0.2, 0.5]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'Batch Loss1:  {l1.item():.4f}')
print(f'Batch Loss2: {l2.item():.4f}')

#----------------
# predicciones
#-----------------
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y}, Y_pred1: {predictions1}, Y_pred2: {predictions2}')

#----------------------------
# Clasificación binaria
#----------------------------
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)  
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoide en el final
        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()

#-----------------------
#Problema Multiclass 
#-----------------------
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax en el final
        return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()  # (Softmax)

