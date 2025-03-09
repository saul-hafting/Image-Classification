import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix
import numpy as np

# Define the Neural Network
class NeuralNet(nn.Module):
    """
    A simple feedforward neural network for MNIST digit classification.
    - Input: 28x28 image (flattened to 784)
    - Hidden Layers: 128, 64 neurons
    - Output: 10 classes (digits 0-9)
    """
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of the neural network."""
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) # No activation on the output layer
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = NeuralNet().to(device)
model_path = "mnist_model.pth"

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Check if a saved model exists
model_path = "mnist_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Loaded trained model. Skipping training.")
else:
    print("No model found. Training from scratch.")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # Save trained model
    torch.save(model.state_dict(), model_path)
    print("Model saved successfully!")


# Evaluate on Test Data
correct, total = 0, 0
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Function to predict custom images
def predict_custom_image(image_path, model):
    """Loads an image, preprocesses it, and predicts the digit."""
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((28, 28)), 
        transforms.ToTensor(),      
        transforms.Normalize((0.5,), (0.5,))  
    ])
    img_tensor = transform(img).unsqueeze(0) 
    model.eval() 
    with torch.no_grad():
        output = model(img_tensor.to(device))
        _, predicted = torch.max(output, 1)
    print(f"Your custom image predicted Digit is: {predicted.item()}")

# Test custom image
predict_custom_image("my_digit.png", model)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("MNIST Confusion Matrix")
plt.show()