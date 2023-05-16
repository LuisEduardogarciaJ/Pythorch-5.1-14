#---------------------------------
#Garcia Junquera Luis Eduardo
#12.1
#----------------------------------

#---------------------------------------------
#Producción = w*x + b
#Producción = funcion_activacion(output)
#----------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-1.0, 1.0, 2.0, 3.0])
#----------
# sofmax
#------------
output = torch.softmax(x, dim=0)
print(output)
sm = nn.Softmax(dim=0)
output = sm(x)
print(output)

#-------------
# sigmoid 
#-------------
output = torch.sigmoid(x)
print(output)
s = nn.Sigmoid()
output = s(x)
print(output)

#--------
# tanh
#--------
output = torch.tanh(x)
print(output)
t = nn.Tanh()
output = t(x)
print(output)
#-----------------
# relu
#-----------------
output = torch.relu(x)
print(output)
relu = nn.ReLU()
output = relu(x)
print(output)
#-----------------
# leaky relu
#---------------
output = F.leaky_relu(x)
print(output)
lrelu = nn.LeakyReLU()
output = lrelu(x)
print(output)
#-------------------------------------------------------------------------------------------------
#nn. ReLU() crea un nn. Módulo que puede agregar, por ejemplo, a un nn. Modelo secuencial.
#torch.relu en el otro lado es solo la llamada API funcional a la función relu,
#so que puede agregarlo, por ejemplo, en su método de avance usted mismo.
#----------------------------------------------------------------------------------------------------

#---------------------------------
# Opción 1 (Crear módulos NN)
#---------------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

#--------------------------------------------------------------------------
#Opcion 2 (Usar activacion de funciones directamente en la evaluacion )
#---------------------------------------------------------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out
    
    