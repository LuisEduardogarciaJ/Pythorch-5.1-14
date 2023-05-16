#----------------------------------
#Garcia Junquera Luis Eduardo
#5.2
#-----------------------------------
import torch
#------------------------
#Calcular el gradiente 

#Regresion lineal
#f = w*x


#ejemplo : f = 2*x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
W = torch.tensor(0.0, dtype= torch.float32, requires_grad=True)


w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#---------
# modelo
#----------
def forward(x):
    return w * x

#--------------------
# error : Perdida = MSE
#---------------------
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

print(f'Prediccion antes del aprendizaje: f(5) = {forward(5).item():.3f}')

#---------------
# aprendizaje
#----------------
learning_rate = 0.01
n_iters = 100
#---------------------------------------------
for epoch in range(n_iters):
    y_pred = forward(X)
    #calcular error 
    l = loss(Y, y_pred)
    #calcular gradiente
    l.backward()
    #mejorar coeficiente
    with torch.no_grad():
        w -= learning_rate * w.grad
        #resetear gradiente
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.8f}')

print(f'Prediccion despues del aprendizaje: f(5) = {forward(5).item():.3f}')