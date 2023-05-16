#-----------------------------------
#Garcia Junquera Luis Eduardo
#13
#-----------------------------------


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#-------------------------
# Configuracion al GPU
#-------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-------------------------
# Hyper-parametros
#-------------------------
input_size = 784 # imagen 28x28
hidden_size = 500 #neuronas ocultas
num_classes = 10  #clasificaciones
num_epochs = 2   #iteraciones sobre los datos 
batch_size = 100 #tamaño de conjunto de datos
learning_rate = 0.001 #tasa de aprendizaje(para que se valla con calma)

#----------------------
# MNIST base de datos
#-----------------------
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

#--------------------
# Cargar datos
#--------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

examples = iter(test_loader) #iterable
example_data, example_targets = next(examples) #siguiente elemento

#-------------------------------------------
#mostar datos en una imagen
#-------------------------------------------
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()

#--------------------------------------------------------------
# Red neuronal completamente conectada para una capa oculta
#--------------------------------------------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # sin activacion y sin softmax al final
        return out

#------------------------------
#Correr modelo en el GPU
#------------------------------
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

#--------------------------------------
# Optimizacion y calculo del error
#---------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

#---------------------------
# Entrenar el modelo
#----------------------------
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # dimension original: [100, 1, 28, 28]
        # Nuevas dimensiones: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        #---------------------------
        #Evaluacion 
        #-----------------------------
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        #------------------------------------------
        # calcculo del gradiente y optimizador
        #--------------------------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
#--------------------------------------------------------------
# Probar el modelo
# En fase de prueba, non requerimos calñcular gradientes
#--------------------------------------------------------------
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max regresa (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
#---------------------------------------------------
#Presicion
#---------------------------------------------------
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')