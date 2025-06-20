# Fashion MNIST Classification using PyTorch

This project implements a simple neural network in PyTorch to classify images from the Fashion MNIST dataset. It includes data loading, model training, evaluation, and visualization of predictions.

---

## 📦 Requirements

Install the necessary libraries using:

```bash
pip install torch torchvision matplotlib
🚀 Run the Code
Copy the following Python code into a file named fashion_mnist_pytorch.py and run it:

Python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Step 1: Define data transformations (normalize to mean=0.5, std=0.5)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Step 2: Load the Fashion MNIST dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

# Step 3: Define the neural network model
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 4: Initialize model, loss function, and optimizer
model = FashionMNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# Step 6: Evaluate model on test set
model.eval()
correct = 0
total = 0
predicted_all = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_all.extend(predicted.numpy())

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 7: Visualize sample predictions
classes = train_dataset.classes
for i in range(5):
    image, label = test_dataset[i]
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, pred = torch.max(output, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Predicted: {classes[pred]}, Actual: {classes[label]}")
    plt.axis('off')
    plt.show()
📊 Output Example
After training, you'll see output similar to this:

Epoch 1/10, Loss: 89.1234
...
Test Accuracy: 86.45%
This will be followed by 5 plots, each comparing a predicted label with its actual label.

🧠 Model Summary
Input: 28x28 grayscale images
Architecture:
Flatten Layer
Dense Layer (128 neurons, ReLU activation)
Dense Layer (10 output classes)
Loss: CrossEntropyLoss
Optimizer: Adam
Epochs: 10
Batch Size: 200
🔧 Suggestions for Improvement
Add Dropout or BatchNorm layers to the model for better generalization.
Consider replacing the simple dense network with a Convolutional Neural Network (CNN) for potentially better performance on image classification tasks.
Utilize GPU acceleration by moving the model and data to a CUDA device (e.g., model.to(device)) if available.
🙌 Acknowledgements
Dataset: Zalando's Fashion MNIST
Framework: PyTorch
