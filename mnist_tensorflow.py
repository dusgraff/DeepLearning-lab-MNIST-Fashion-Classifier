# Import required libraries
from utils.writehost import *
from colorama import Fore, Back
import tensorflow as tf
import os
import multiprocessing
from mnist_tensorflow_model_class import MnistFashionCnnModel

# Get number of CPU cores
num_cores = multiprocessing.cpu_count()

# Configure TensorFlow to use all CPU cores
tf.config.threading.set_inter_op_parallelism_threads(num_cores)
tf.config.threading.set_intra_op_parallelism_threads(num_cores)

# Enable memory growth
physical_devices = tf.config.list_physical_devices('CPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Optimize for CPU
tf.config.optimizer.set_jit(True)  # Enable XLA compilation

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR
tf.get_logger().setLevel('ERROR')  # Only show errors, not warnings

clear_screen()
print_header("Importing Modules")



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

# -------------------------------------------------------------------------------
print_header("Loading Data") 

# Create instance of CNNModel with specific learning rate
model = MnistFashionCnnModel(learning_rate=0.0005)

# Load data into model
model.load_data(
    train_data_path=train_data_path,
    train_labels_path=train_labels_path,
    test_data_path=test_data_path,
    test_labels_path=test_labels_path
)

# Print confirmation message and data shapes
train_data, train_labels, test_data, test_labels = model.get_data()
print("\nFashion MNIST dataset loaded from .\\datasets\\")
print(f"Training data shape:   {Fore.GREEN}{str(train_data.shape):<20}{Fore.RESET}")
print(f"Training labels shape: {Fore.GREEN}{str(train_labels.shape):<20}{Fore.RESET}")
print(f"Test data shape:       {Fore.GREEN}{str(test_data.shape):<20}{Fore.RESET}")
print(f"Test labels shape:     {Fore.GREEN}{str(test_labels.shape):<20}{Fore.RESET}")

# -------------------------------------------------------------------------------
print_header("Initial Model Evaluation") 

# Evaluate initial model performance on both datasets
train_loss, train_accuracy = model.evaluate(dataset='train', batch_size=64)
test_loss, test_accuracy = model.evaluate(dataset='test', batch_size=64)

# Print initial metrics
print("\nInitial Model Performance:")
print(f"Training Loss:       {Fore.GREEN}{train_loss:>15.4f}{Fore.RESET}")
print(f"Test Loss:          {Fore.GREEN}{test_loss:>15.4f}{Fore.RESET}")
print(f"Training Accuracy:   {Fore.GREEN}{train_accuracy:>15.2f}%{Fore.RESET}")
print(f"Test Accuracy:      {Fore.GREEN}{test_accuracy:>15.2f}%{Fore.RESET}")

# -------------------------------------------------------------------------------
print_header("Training Model")

# Initial training phase
print("\nPhase 1: Initial Training (5 epochs)")
history1 = model.train(epochs=20, batch_size=64, early_drop_out=True, verbose=True)


# Save model parameters
model.save_parameters('.\\checkpoints\\mnist_model')




