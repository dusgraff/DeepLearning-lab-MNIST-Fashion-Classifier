# Import writehost module from utils package
from utils.writehost import *
from colorama import Fore, Back

clear_screen()
print_header("Importing Modules")

# Import required libraries for downloading Fashion MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import time
import sys

# Print header for imported modules
print("\nImports complete")

# -------------------------------------------------------------------------------
print_header("Path Definitions")    

# path definitions
data_dir = '.\\datasets'

train_data_path = f'{data_dir}\\train_data.npy'
train_labels_path = f'{data_dir}\\train_labels.npy'
test_data_path = f'{data_dir}\\test_data.npy'
test_labels_path = f'{data_dir}\\test_labels.npy'

# Print confirmation message and data shapes
print(f"Training data path:   {Fore.MAGENTA}{train_data_path:>35}{Fore.RESET}")
print(f"Training labels path: {Fore.MAGENTA}{train_labels_path:>35}{Fore.RESET}")
print(f"Test data path:       {Fore.MAGENTA}{test_data_path:>35}{Fore.RESET}")
print(f"Test labels path:     {Fore.MAGENTA}{test_labels_path:>35}{Fore.RESET}")

# Define the class names for Fashion MNIST labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# -------------------------------------------------------------------------------
print_header("Load MNIST Dataset") 

# Load training data and labels from numpy files
train_data = np.load(train_data_path)
train_labels = np.load(train_labels_path)

# Load test data and labels from numpy files
test_data = np.load(test_data_path)
test_labels = np.load(test_labels_path)

# Print confirmation message and data shapes
print("Fashion MNIST dataset loaded from .\\datasets\\")
print(f"Training data shape:   {Fore.GREEN}{str(train_data.shape):<20}{Fore.RESET}")
print(f"Training labels shape: {Fore.GREEN}{str(train_labels.shape):<20}{Fore.RESET}")
print(f"Test data shape:       {Fore.GREEN}{str(test_data.shape):<20}{Fore.RESET}")
print(f"Test labels shape:     {Fore.GREEN}{str(test_labels.shape):<20}{Fore.RESET}")



# -------------------------------------------------------------------------------
print_header("Normalizing Data") 

# Convert numpy arrays to PyTorch tensors and normalize pixel values to [0,1]
train_data = torch.FloatTensor(train_data) / 255.0 
train_labels = torch.LongTensor(train_labels)
test_data = torch.FloatTensor(test_data) / 255.0
test_labels = torch.LongTensor(test_labels)

# Add channel dimension for CNN input (N, H, W) -> (N, C, H, W)
train_data = train_data.unsqueeze(1)
test_data = test_data.unsqueeze(1)

# Print confirmation message and data shapes
print("Fashion MNIST dataset loaded from .\\datasets\\")
print(f"Training data shape:   {Fore.GREEN}{str(train_data.shape):<20}{Fore.RESET}")
print(f"Training labels shape: {Fore.GREEN}{str(train_labels.shape):<20}{Fore.RESET}")
print(f"Test data shape:       {Fore.GREEN}{str(test_data.shape):<20}{Fore.RESET}")
print(f"Test labels shape:     {Fore.GREEN}{str(test_labels.shape):<20}{Fore.RESET}")

# -------------------------------------------------------------------------------
print_header("Defining the Model")

# Define the model architecture
# -------------------------------------------------------------------------------
# Description: 
#     CNN model for MNIST classification with 2 convolutional layers followed by 3 fully connected layers
#     Input shape: (batch_size, 1, 28, 28)
#     Output shape: (batch_size, 10) - one score per class

# Example:
#     model = CNN()
#     output = model(input_tensor) # input_tensor shape: (batch_size, 1, 28, 28)

class CNN(nn.Module):
    def __init__(self):
        # Call parent class constructor
        super(CNN, self).__init__()
        
        # Define first convolutional layer
        # Input: 1 channel (grayscale), Output: 32 feature maps, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # Define second convolutional layer
        # Input: 32 channels, Output: 64 feature maps, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Define max pooling layer (2x2 window)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Define first fully connected layer
        # Input: 64 feature maps * 7 * 7 (after 2 pooling operations), Output: 512 neurons
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        
        # Define second fully connected layer
        # Input: 512 neurons, Output: 128 neurons
        self.fc2 = nn.Linear(512, 128)
        
        # Define output layer
        # Input: 128 neurons, Output: 10 classes
        self.fc3 = nn.Linear(128, 10)
        
        # Define dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Apply first conv layer followed by ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second conv layer followed by ReLU and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the feature maps
        x = x.view(-1, 64 * 7 * 7)
        
        # Apply first fully connected layer with ReLU and dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Apply second fully connected layer with ReLU and dropout
        x = self.dropout(F.relu(self.fc2(x)))
        
        # Apply output layer
        x = self.fc3(x)
        
        return x

# Create an instance of the model
model = CNN()

# Print model architecture
print(model)

# -------------------------------------------------------------------------------
print_header("Evaluating initial model performance")


# -------------------------------------------------------------------------------
# Description: 
#     Calculates model loss on test dataset
#     Returns average loss value
# Example:
#     test_loss = get_test_loss(model, test_data, test_labels)

def get_test_loss(model, data, labels, batch_size=64):
    # Set model to evaluation mode
    model.eval()
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize total loss
    total_loss = 0.0
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Process data in batches
        for i in range(0, len(data), batch_size):
            # Get batch of data and labels
            inputs = data[i:i+batch_size]
            targets = labels[i:i+batch_size]
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss for this batch
            loss = criterion(outputs, targets)
            
            # Add batch loss to total
            total_loss += loss.item() * len(inputs)
    
    # Calculate average loss
    avg_loss = total_loss / len(data)
    
    # Set model back to training mode
    model.train()
    
    return avg_loss

# Calculate initial test loss
test_loss = get_test_loss(model, test_data, test_labels)

# Print initial model metrics
print(f"Initial test loss:     {Fore.GREEN}{test_loss:>15.4f}{Fore.RESET}")



# -------------------------------------------------------------------------------
# Description: 
#     Evaluates model performance on given dataset and labels
#     Returns accuracy and predictions
# Example:
#     accuracy, predictions = evaluate_model(model, test_data, test_labels)

def evaluate_model(model, data, labels, batch_size=64):
    # Set model to evaluation mode
    model.eval()
    
    # Initialize counters for accuracy calculation
    correct = 0
    total = 0
    
    # List to store predictions
    predictions = []
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Process data in batches
        for i in range(0, len(data), batch_size):
            # Get batch of data and labels
            inputs = data[i:i+batch_size]
            targets = labels[i:i+batch_size]
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            
            # Update accuracy counters
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    # Calculate accuracy
    accuracy = 100 * correct / total

    # Set model back to training mode
    model.train()

    return accuracy, predictions

# Evaluate initial model performance on training data
train_accuracy, train_predictions = evaluate_model(model, train_data, train_labels)

# Print initial model performance
print(f"Initial Training Accuracy: {Fore.GREEN}{train_accuracy:>15.2f}%{Fore.RESET}")


# -------------------------------------------------------------------------------
print_header("Training Configuration")

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reduced learning rate

# Set training hyperparameters
epochs = 20  # Increased epochs
batch_size = 64  # Reduced batch size

# Set model back to training mode
model.train()


# Print training configuration details
print(f"Loss Function:        {Fore.GREEN}{'CrossEntropyLoss':>15}{Fore.RESET}")
print(f"Optimizer:           {Fore.GREEN}{'Adam':>15}{Fore.RESET}")
print(f"Learning Rate:       {Fore.GREEN}{'0.0005':>15}{Fore.RESET}")
print(f"Number of Epochs:    {Fore.GREEN}{epochs:>15}{Fore.RESET}")
print(f"Batch Size:          {Fore.GREEN}{batch_size:>15}{Fore.RESET}")
print(f"Training Samples:    {Fore.GREEN}{len(train_data):>15}{Fore.RESET}")
print(f"Validation Samples:  {Fore.GREEN}{len(test_data):>15}{Fore.RESET}")


# -------------------------------------------------------------------------------
print_header("Training the Model")

# Train the model
for epoch in range(epochs):
    start_time = time.time()
    print(f"Epoch {epoch+1} started at {time.strftime('%H:%M:%S', time.gmtime(start_time))}")

    for i in range(0, len(train_data), batch_size):
        inputs, labels = train_data[i:i+batch_size], train_labels[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()


    print(f"Epoch {epoch+1} completed at {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
    print(f"Time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    print(f"Loss: {loss.item()}")

    # Evaluate model performance on training data
    train_accuracy, train_predictions = evaluate_model(model, train_data, train_labels)
    print(f"Training Accuracy: {Fore.GREEN}{train_accuracy:>15.2f}%{Fore.RESET}")
    
    # Get test accuracy and loss using evaluate_model function
    test_accuracy, test_predictions = evaluate_model(model, test_data, test_labels)
    
    # Get current test loss
    test_loss = get_test_loss(model, test_data, test_labels)
    
    # Print training and test metrics comparison
    print(f"Training Loss:       {Fore.GREEN}{loss.item():>15.4f}{Fore.RESET}")
    print(f"Test Loss:          {Fore.GREEN}{test_loss:>15.4f}{Fore.RESET}")
    print(f"Training Accuracy:   {Fore.GREEN}{train_accuracy:>15.2f}%{Fore.RESET}")
    print(f"Test Accuracy:      {Fore.GREEN}{test_accuracy:>15.2f}%{Fore.RESET}")
    print("------------------------------------------------")
    
    # Early stopping check - if test loss is significantly higher than training loss
    if test_loss > 1.5 * loss.item():
        print(f"{Fore.RED}Early stopping triggered - model may be overfitting{Fore.RESET}")
        break


    

# -------------------------------------------------------------------------------
print_header("Evaluating the Model")

# Initialize counters for accuracy calculation
correct = 0
total = 0

# Validation loop
with torch.no_grad():  # Disable gradient calculation for validation
    for i in range(0, len(test_data), batch_size):
        inputs, labels = test_data[i:i+batch_size], test_labels[i:i+batch_size]
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)  # Add batch size to total
        correct += (predicted == labels).sum().item()  # Add number of correct predictions

# Calculate accuracy
accuracy = 100 * correct / total

print(f"Accuracy: {accuracy:.2f}%")

