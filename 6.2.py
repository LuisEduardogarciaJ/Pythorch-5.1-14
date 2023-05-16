#----
#6.2
#------

# 1) Modelo de diseño (entrada, salida, pase hacia adelante con diferentes capas)
# 2) Pérdida de construcción y optimizador
# 3) Bucle de entrenamiento
# - Adelante = cálculo de predicción y pérdida
# - Hacia atrás = calcular gradientes
# - Actualizar pesos

import torch
import torch.nn as nn

#-------------------------------------------
# Regresion lineal 
# f = w * x 

# f = 2 * x
#-----------------------------------------------
# 0) 
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'#samples: {n_samples}, #features: {n_features}')
# 0) create a test sample
X_test = torch.tensor([5], dtype=torch.float32)

# 1) Design Model, the model has to implement the forward pass!
# Here we can use a built-in model from PyTorch
input_size = n_features
output_size = n_features

# we can call this model with samples X
model = nn.Linear(input_size, output_size)

'''
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
'''

'''
clase LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regresión Lineal, auto).__init__()
        # definir diferentes capas
        self.lin = nn.Linear(entrada_dim, salida_dim)

    def adelante(auto, x):
        volver self.lin(x)

modelo = Regresión lineal (tamaño de entrada, tamaño de salida)
'''

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# 2) Define loss and optimizer
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#--------------------------
# 3) Ciclo de parendizaje
#--------------------------
for epoch in range(n_iters):
    # predict = forward pass with our model
    y_predicted = model(X)

    # loss
    l = loss(Y, y_predicted)

    # calcular el gradiente = retropropagacion(backward)
    l.backward()
    # mejorar coeficiente
    optimizer.step()

    # resetear coeficiente
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters() #Parametros
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l)

print(f'Prediccion despues del parendizaje: f(5) = {model(X_test).item():.3f}')