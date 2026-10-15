from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
MODEL_PATH=Path(__file__).resolve().with_name("resnet18_cifar10.pth")
classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
model=models.resnet18(weights=None)
model.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
model.maxpool=nn.Identity()
model.fc=nn.Linear(model.fc.in_features,10)
checkpoint=torch.load(MODEL_PATH,map_location="cpu",weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),
                              transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
root=Tk()
root.withdraw()
path=askopenfilename(filetypes=[("Image Files","*.jpg *.jpeg *.png *.bmp")])
root.destroy()
image=Image.open(path).convert("RGB")
input_image=transform(image).unsqueeze(0)
with torch.no_grad():
    output=model(input_image)
    probability=torch.softmax(output,1)
    confidence,prediction=torch.max(probability,1)
predicted_class=classes[prediction.item()]
confidence=confidence.item()*100
plt.figure(figsize=(7,7))
plt.imshow(image)
plt.axis("off")
plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%",fontsize=14)
plt.show()