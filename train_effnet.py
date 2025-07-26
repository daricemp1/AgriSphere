
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os

# Paths
data_dir = "sample_data/images"
batch_size = 32
epochs = 5
model_out = "efficientnet_disease.pt"
labels_out = "class_names.pt"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Dataset and loader
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
class_names = dataset.classes

# Model
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

model.train()
for epoch in range(epochs):
    running_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}")

# Save model and labels
torch.save(model.state_dict(), model_out)
torch.save(class_names, labels_out)
print(f"✅ Model saved to {model_out}")
print(f"✅ Labels saved to {labels_out}")
