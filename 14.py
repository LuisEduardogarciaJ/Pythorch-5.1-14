#-----------------------------------
#Gracia Junquera Luis Eduardo
#14
#Ejemplo de red neuronal computacional
#--------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#------------------------
#Configuarcion del GPU
#-------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#--------------------
# Hyper-parametros 
#-------------------
num_epochs = 5  #ITERACIONES SOBRE LOS DATOS DE ENTRENAMIENTO
batch_size = 4 #Subconjntos de datos
learning_rate = 0.001 #Tasa de aprendizaje

#---------------------------------------------------------------
#Definir pre- procesamientos de datos8transformacion)
# conjunto de datos tiene imágenes PILImage de rango [0, 1].
# Los transformamos a Tensores de rango normalizado [-1, 1]
#----------------------------------------------------------------
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#---------------------------------------------------------------------------------------
# CIFAR10: 60000 imágenes a color de 32x32 en 10 clases, con 6000 imágenes por clase
#----------------------------------------------------------------------------------------
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')#---------------------------avion, auto, pajaro,gato,perro,rana,caballo,barco

def imshow(img):
    img = img / 2 + 0.5  # desnormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#------------------------------------------------------
# obtener algunas imágenes de entrenamiento al azar
#-------------------------------------------------------
dataiter = iter(train_loader)
images, labels = next(dataiter)

# ----------------------------------------------
#Imprime las imagenes
#-----------------------------------------------
imshow(torchvision.utils.make_grid(images))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #------------------------------------------------------------------------------------
       # forma de origen: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 canales de entrada, 6 canales de salida, 5 tamaños de kernel
        #-------------------------------------------------------------------------------------
        images = images.to(device)
        labels = labels.to(device)

#----------------------------------
        # Pase adelantado
        #--------------------------
        outputs = model(images)
        loss = criterion(outputs, labels)
        #--------------------------
        # Retroceder y optimizar
        #--------------------------
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # Regresa los maximos (evalua)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
