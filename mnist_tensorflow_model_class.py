# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from colorama import Fore
import numpy as np
import os

# -------------------------------------------------------------------------------
# Description: 
#     CNN model for MNIST classification with 2 convolutional layers followed by 3 fully connected layers
#     Input shape: (batch_size, 28, 28, 1)
#     Output shape: (batch_size, 10) - one score per class

# Example:
#     model = CNNModel()
#     output = model(input_tensor) # input_tensor shape: (batch_size, 28, 28, 1)

class MnistFashionCnnModel(tf.keras.Model):

    # -------------------------------------------------------------------------------
    # Description: 
    #     Constructor for CNNModel class that initializes all layers
    #     Sets up convolutional, pooling, dense and dropout layers
    #     Layer architecture matches PyTorch implementation for consistency
    
    # Example:
    #     model = CNNModel()
    #     model.build((None, 28, 28, 1))  # Initialize model weights

    def __init__(self, learning_rate=0.0005):
        # Call parent class constructor
        super(MnistFashionCnnModel, self).__init__()
        
        # Set random seed for reproducibility
        tf.random.set_seed(112358)
        
        # Initialize private data containers
        self._train_data = None
        self._train_labels = None
        self._test_data = None
        self._test_labels = None
        
        # Define layers with kernel initializers for random weights
        self.conv1 = layers.Conv2D(32, (3, 3), padding='same',
                                 kernel_initializer=tf.keras.initializers.HeNormal(seed=112358))
        
        self.conv2 = layers.Conv2D(64, (3, 3), padding='same',
                                 kernel_initializer=tf.keras.initializers.HeNormal(seed=112358))
        
        self.pool = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        
        self.fc1 = layers.Dense(512,
                              kernel_initializer=tf.keras.initializers.HeNormal(seed=112358))
        
        self.fc2 = layers.Dense(128,
                              kernel_initializer=tf.keras.initializers.HeNormal(seed=112358))
        
        self.fc3 = layers.Dense(10,
                              kernel_initializer=tf.keras.initializers.HeNormal(seed=112358))
        
        self.dropout = layers.Dropout(0.5, seed=112358)
        
        # Define loss function and optimizer
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # -------------------------------------------------------------------------------
    # Description: 
    #     Forward pass through the CNN model
    #     Applies convolutional, pooling, dense layers with ReLU activations and dropout
    #     Input shape: (batch_size, 28, 28, 1)
    #     Output shape: (batch_size, 10) - one score per class
    
    # Example:
    #     model = CNNModel()
    #     output = model(input_tensor, training=True) # For training
    #     predictions = model(input_tensor, training=False) # For inference

    def call(self, x, training=False):
        # Apply first conv layer followed by ReLU and pooling
        x = self.pool(tf.nn.relu(self.conv1(x)))
        
        # Apply second conv layer followed by ReLU and pooling
        x = self.pool(tf.nn.relu(self.conv2(x)))
        
        # Flatten the feature maps
        x = self.flatten(x)
        
        # Apply first fully connected layer with ReLU and dropout
        x = self.dropout(tf.nn.relu(self.fc1(x)), training=training)
        
        # Apply second fully connected layer with ReLU and dropout
        x = self.dropout(tf.nn.relu(self.fc2(x)), training=training)
        
        # Apply output layer
        x = self.fc3(x)
        
        return x
    
    
    # -------------------------------------------------------------------------------
    # Description: 
    #     Load and preprocess MNIST data from numpy files
    #     Stores processed data in class variables
    #
    # Example:
    #     model.load_data(train_data_path='./datasets/train_data.npy')

    def load_data(self, train_data_path, train_labels_path, test_data_path, test_labels_path):
        try:
            # Load and process data
            train_data = np.load(train_data_path)
            train_labels = np.load(train_labels_path)
            test_data = np.load(test_data_path)
            test_labels = np.load(test_labels_path)
            
            # Set the processed data
            self.set_data(train_data, train_labels, test_data, test_labels)
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find one of the data files: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    # -------------------------------------------------------------------------------
    # Description: 
    #     Set the model's data manually
    #     Handles preprocessing of raw data
    #
    # Example:
    #     model.set_data(train_data, train_labels, test_data, test_labels)

    def set_data(self, train_data, train_labels, test_data, test_labels):
        # Normalize pixel values to [0,1]
        self._train_data = train_data.astype('float32') / 255.0
        self._test_data = test_data.astype('float32') / 255.0
        
        # Convert labels to int64 type
        self._train_labels = train_labels.astype('int64')
        self._test_labels = test_labels.astype('int64')
        
        # Add channel dimension for CNN input
        self._train_data = self._train_data[..., tf.newaxis]
        self._test_data = self._test_data[..., tf.newaxis]

    # -------------------------------------------------------------------------------
    # Description: 
    #     Get the model's data
    #     Returns tuple of (train_data, train_labels, test_data, test_labels)
    #
    # Example:
    #     train_data, train_labels, test_data, test_labels = model.get_data()

    def get_data(self):
        return self._train_data, self._train_labels, self._test_data, self._test_labels

    # -------------------------------------------------------------------------------
    # Description:
    #     Evaluates model performance on training or test set
    #
    # Example:
    #     loss, accuracy = model.evaluate(dataset='test')

    def evaluate(self, batch_size=32, dataset='test', verbose=True, return_predictions=False):
        
        # Select appropriate dataset
        if dataset.lower() == 'train':
            data = self._train_data
            labels = self._train_labels
            
        else:
            data = self._test_data
            labels = self._test_labels
            
        # Create TensorFlow dataset from input data
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        
        # Batch and prefetch the dataset for evaluation
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Initialize metrics to track loss and accuracy
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        
        # Progress tracking
        num_batches = len(data) // batch_size + (1 if len(data) % batch_size else 0)
        
        # Iterate through batches
        for batch_idx, (batch_data, batch_labels) in enumerate(dataset):
            # Get model predictions
            logits = self(batch_data, training=False)
            
            # Calculate loss using sparse categorical crossentropy
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    batch_labels, logits, from_logits=True
                )
            )
            
            # Get predicted classes
            predictions = tf.argmax(logits, axis=1)
            
            if return_predictions:
                all_predictions.extend(predictions.numpy())
            
            # Calculate number of correct predictions
            correct = tf.equal(predictions, batch_labels)
            correct = tf.reduce_sum(tf.cast(correct, tf.float32))
            
            # Update running totals
            total_loss += loss * len(batch_data)
            total_correct += correct
            total_samples += len(batch_data)
            
            # Print progress if verbose
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"\rEvaluating: {batch_idx + 1}/{num_batches} batches", end="")
        
        if verbose:
            print()  # New line after progress
        
        # Calculate final metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        if return_predictions:
            return avg_loss.numpy(), accuracy.numpy(), all_predictions
        
        return avg_loss.numpy(), accuracy.numpy()
    
    # -------------------------------------------------------------------------------
    # Description: 
    #     Train the model using specified dataset
    #     Defaults to training data
    #
    # Example:
    #     history = model.train(epochs=20)
    #     history = model.train(epochs=20, dataset='train')  # explicit

    def train(self, epochs=3, batch_size=32, early_drop_out=True, dataset='train', verbose=True, return_predictions=False):
        # Select appropriate dataset
        if dataset.lower() == 'test':
            data = self._test_data
            labels = self._test_labels
        else:
            data = self._train_data
            labels = self._train_labels
        
        if data is None or labels is None:
            raise ValueError(f"No {dataset} data available. Call load_data() or set_data() first.")
        
        # Optimize dataset pipeline
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        dataset = dataset.cache()  # Cache the dataset in memory
        dataset = dataset.shuffle(buffer_size=1000)  # Shuffle with larger buffer
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch next batch
        
        # Initialize history tracking
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Training loop over epochs
        for epoch in range(epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Initialize metrics for this epoch
            total_loss = 0
            total_correct = 0
            total_samples = 0
            all_predictions = []
            
            # Progress tracking
            num_batches = len(data) // batch_size + (1 if len(data) % batch_size else 0)
            
            # Iterate through batches
            for batch_idx, (batch_data, batch_labels) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    logits = self(batch_data, training=True)
                    loss = self.loss_fn(batch_labels, logits)
                
                # Backpropagation
                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                
                # Get predictions
                predictions = tf.argmax(logits, axis=1)
                if return_predictions:
                    all_predictions.extend(predictions.numpy())
                
                # Calculate accuracy
                correct = tf.equal(predictions, batch_labels)
                correct = tf.reduce_sum(tf.cast(correct, tf.float32))
                
                # Update metrics
                total_loss += loss * len(batch_data)
                total_correct += correct
                total_samples += len(batch_data)
                
                # Print batch progress
                if verbose and (batch_idx + 1) % 10 == 0:
                    print(f"\rTraining: {batch_idx + 1}/{num_batches} batches", end="")
            
            # Calculate epoch metrics
            epoch_loss = total_loss / total_samples
            epoch_accuracy = total_correct / total_samples
            
            # Store training metrics in history
            history['loss'].append(float(epoch_loss))
            history['accuracy'].append(float(epoch_accuracy))
            
            # Early dropout check
            if early_drop_out:
                # Evaluate on test set
                test_loss, test_accuracy = self.evaluate(
                    dataset='test', 
                    batch_size=batch_size, 
                    verbose=False
                )
                
                # Store validation metrics
                history['val_loss'].append(float(test_loss))
                history['val_accuracy'].append(float(test_accuracy))
                
                # Check for overfitting conditions
                if test_loss > 1.5 * epoch_loss:  # Loss gap too large
                    if verbose:
                        print(f"\n{Fore.RED}Early stopping: Test loss ({test_loss:.4f}) much higher than training loss ({epoch_loss:.4f}){Fore.RESET}")
                    break
                    
                if epoch > 0 and test_accuracy < history['val_accuracy'][-2]:  # Accuracy decreasing
                    if verbose:
                        print(f"\n{Fore.RED}Early stopping: Test accuracy decreased from {history['val_accuracy'][-2]:.2%} to {test_accuracy:.2%}{Fore.RESET}")
                    break
            
            # Print epoch results
            if verbose:
                print(f"\nEpoch {epoch + 1}: loss = {epoch_loss:.4f}, accuracy = {epoch_accuracy:.2%}")
                if early_drop_out:
                    print(f"           val_loss = {test_loss:.4f}, val_accuracy = {test_accuracy:.2%}")
        
        if return_predictions:
            return history, all_predictions
        
        return history

    # -------------------------------------------------------------------------------
    # Description: 
    #     Save model parameters to a file
    #     Saves weights, optimizer state, and training history
    #
    # Example:
    #     model.save_parameters('.\\checkpoints\\model_checkpoint')

    def save_parameters(self, checkpoint_path):
        try:
            # Create checkpoint directory if it doesn't exist
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            
            # Create checkpoint object
            checkpoint = tf.train.Checkpoint(
                model=self,
                optimizer=self.optimizer
            )
            
            # Save the checkpoint
            checkpoint.save(checkpoint_path)
            
            if self.verbose:
                print(f"\nModel parameters saved to: {checkpoint_path}")
            
        except Exception as e:
            raise Exception(f"Error saving model parameters: {str(e)}")

    # -------------------------------------------------------------------------------
    # Description: 
    #     Load model parameters from a file
    #     Restores weights and optimizer state
    #
    # Example:
    #     model.load_parameters('.\\checkpoints\\model_checkpoint-1')

    def load_parameters(self, checkpoint_path):
        try:
            # Create checkpoint object
            checkpoint = tf.train.Checkpoint(
                model=self,
                optimizer=self.optimizer
            )
            
            # Restore the checkpoint
            status = checkpoint.restore(checkpoint_path)
            
            # Wait for restore to complete
            status.expect_partial()
            
            if self.verbose:
                print(f"\nModel parameters loaded from: {checkpoint_path}")
            
        except Exception as e:
            raise Exception(f"Error loading model parameters: {str(e)}")




















# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
if __name__ == "__main__":

    # Create model instance
    model = MnistFashionCnnModel()

    # Print model summary
    model.build((None, 28, 28, 1))
    model.summary()

    # Print confirmation message
    print(f"\nModel architecture initialized successfully")
    print(f"Input shape:  {Fore.GREEN}{'(batch, 28, 28, 1)':>20}{Fore.RESET}")
    print(f"Output shape: {Fore.GREEN}{'(batch, 10)':>20}{Fore.RESET}")
