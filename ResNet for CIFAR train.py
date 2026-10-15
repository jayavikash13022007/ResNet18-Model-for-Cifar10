from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
BATCH_SIZE=128;EPOCHS=15;LEARNING_RATE=0.001
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
train_data=datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
test_data=datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)
train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False,pin_memory=True)
model=models.resnet18(weights=None)
model.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
model.maxpool=nn.Identity()
model.fc=nn.Linear(model.fc.in_features,10)
model=model.to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)
train_losses=[]
train_accuracies=[]
for epoch in range(EPOCHS):
    model.train()
    correct=0;total=0;running_loss=0
    for images,labels in train_loader:
        images=images.to(device,non_blocking=True)
        labels=labels.to(device,non_blocking=True)
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    accuracy=100*correct/total
    loss=running_loss/len(train_loader)
    train_losses.append(loss)
    train_accuracies.append(accuracy)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss:.4f} Accuracy: {accuracy:.2f}%")
model.eval()
correct=0;total=0
with torch.no_grad():
    for images,labels in test_loader:
        images=images.to(device,non_blocking=True)
        labels=labels.to(device,non_blocking=True)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
test_accuracy=100*correct/total
print(f"\nTest Accuracy: {test_accuracy:.2f}%")
model_path=Path(__file__).resolve().with_name("resnet18_cifar10.pth")
torch.save({"model_state_dict":model.state_dict()},model_path)
plt.plot(train_accuracies)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet-18 Training Accuracy")
plt.show()
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet-18 Training Loss")
plt.show()