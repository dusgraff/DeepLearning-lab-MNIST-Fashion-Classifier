# Import writehost module from utils package
from utils.writehost import *
from colorama import Fore, Back

clear_screen()
print_header("Importing Modules")

# Import required libraries for downloading Fashion MNIST
import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
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

# clear datasets directory
if os.path.exists('.\\datasets'):  
    print_text("Clearing datasets directory", Fore.RED, 1)
    shutil.rmtree('.\\datasets')

# create datasets directory
os.makedirs('.\\datasets')

# Define the class names for Fashion MNIST labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# -------------------------------------------------------------------------------
print_header("Downloading and Loading MNIST Dataset") 

# Load Fashion MNIST dataset from tensorflow
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Save training data and labels in original format (N, H, W)
np.save(train_data_path, train_data)
np.save(train_labels_path, train_labels)

# Save test data and labels 
np.save(test_data_path, test_data)
np.save(test_labels_path, test_labels)

# Print confirmation message and data shapes
print("Fashion MNIST dataset saved to .\\datasets\\")
print(f"Training data shape:   {Fore.GREEN}{str(train_data.shape):<20}{Fore.RESET}")
print(f"Training labels shape: {Fore.GREEN}{str(train_labels.shape):<20}{Fore.RESET}")
print(f"Test data shape:       {Fore.GREEN}{str(test_data.shape):<20}{Fore.RESET}")
print(f"Test labels shape:     {Fore.GREEN}{str(test_labels.shape):<20}{Fore.RESET}") 



# -------------------------------------------------------------------------------
print_header("Sample Images from MNIST Dataset") 

# Select 3 random indices from the training data
random_indices = np.random.randint(0, len(train_data), 3)

# Create a figure with 3 subplots - size adjusted for 28x28 pixel images with padding
plt.figure(figsize=(4, 1.5))

# Loop through the random indices and plot each image
for i, idx in enumerate(random_indices):
    # Create subplot
    plt.subplot(1, 3, i+1)
    
    # Display the image in grayscale without interpolation to maintain pixel clarity
    plt.imshow(train_data[idx], cmap='gray', interpolation='nearest')
    
    # Add title with the class name instead of numeric label
    plt.title(f'{class_names[train_labels[idx]]}', fontsize=8)
    
    # Remove axes
    plt.axis('off')

# Adjust layout with minimal spacing between subplots
plt.subplots_adjust(wspace=0.3)
plt.show()
