# -*- coding: utf-8 -*-
"""
Deepfake Face Detection using Parameter-Tuned CNN, tf.data pipeline,
Mixed Precision, and CSV Logging.

This script uses a single fine-tuned CNN instead of an ensemble approach
with an efficient tf.data input pipeline and mixed precision training.
It classifies faces as 'real' or 'fake' and logs metrics to a CSV file.

Dataset Structure Expectation:
-----------------------------
dataset_root/
├── train/
│   ├── real/ ...
│   └── fake/ ...
├── validation/
│   ├── real/ ...
│   └── fake/ ...
└── test/      # Optional
    ├── real/ ...
    └── fake/ ...

Instructions:
------------
1.  Modify `DATASET_DIR` variable.
2.  Adjust `BATCH_SIZE`, `EPOCHS` if needed.
3.  Run sequentially. Training logs and model weights will be saved.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
# Import layers for data augmentation
from tensorflow.keras import layers
# Import necessary components for the CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# Import Mixed Precision API
from tensorflow.keras import mixed_precision
# Import Callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import time # To potentially time data loading

# Import tqdm for progress bars
try:
    from tqdm import tqdm
    from tqdm.auto import tqdm as tqdm_auto
    from tqdm.keras import TqdmCallback
    TQDM_AVAILABLE = True
except ImportError:
    print("Installing tqdm package for better progress visualization...")
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm"])
    from tqdm import tqdm
    from tqdm.auto import tqdm as tqdm_auto
    from tqdm.keras import TqdmCallback
    TQDM_AVAILABLE = True

# --- Configuration ---
DATASET_DIR = 'Dataset/' # Path to your merged dataset
TRAIN_DIR = os.path.join(DATASET_DIR, 'Train')
VAL_DIR = os.path.join(DATASET_DIR, 'Validation')
TEST_DIR = os.path.join(DATASET_DIR, 'Test') 

IMG_HEIGHT = 164  # Higher resolution for better feature extraction
IMG_WIDTH = 164   
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 16    # Batch size for 224x224 images
EPOCHS = 50       # Number of training epochs
LEARNING_RATE = 0.00005
AUTOTUNE = tf.data.AUTOTUNE

# New configuration for ensemble model
USE_CLASS_WEIGHTS = True   # Set to True to use class weights if imbalance detected

CSV_LOG_FILE = 'training_log_tuned_cnn.csv'

# --- Enable Mixed Precision ---
# Use float16 for computations where possible for speed and memory efficiency
# Should be done at the start of the script
print("Enabling Mixed Precision Training (mixed_float16)")
mixed_precision.set_global_policy('mixed_float16')

# --- Optimize GPU Memory Usage ---
print("Optimizing GPU memory usage...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Memory growth needs to be the same across all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled on {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"Memory growth setting error: {e}")

# --- Data Loading and Preprocessing with tf.data ---

print("Setting up tf.data pipelines...")

# Define enhanced data augmentation layers for 128x128 images
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.2),               # Increased rotation
    layers.RandomZoom(0.2),                   # Increased zoom variation
    layers.RandomBrightness(0.2),             # Increased brightness variation
    layers.RandomContrast(0.2),               # Increased contrast variation
    layers.RandomFlip("horizontal"),          # Horizontal flip
    layers.RandomFlip("vertical"),            # Added vertical flip 
    layers.GaussianNoise(0.1),                # Increased noise to improve robustness
    layers.RandomTranslation(0.1, 0.1),       # Added random translation
], name="data_augmentation")

# Function to load and preprocess datasets using tf.keras.utils with tqdm progress bar
def create_dataset(directory, subset=None, validation_split=None, shuffle=True):
    if not directory or not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return None
    try:
        print(f"Loading dataset from {directory}...")
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels='inferred',
            label_mode='binary',
            image_size=IMAGE_SIZE,
            interpolation='nearest',
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            seed=123,
            validation_split=validation_split,
            subset=subset
        )
        if ds is None:
             print(f"Failed to load dataset from {directory}")
             return None

        # Get class names before mapping (mapping removes this attribute)
        class_names = ds.class_names
        print(f"Found classes in {directory}: {class_names}")
        if len(class_names) != 2:
             raise ValueError(f"Expected 2 classes (real/fake), but found {len(class_names)} in {directory}")

        return ds, class_names

    except Exception as e:
        print(f"Error loading dataset from {directory}: {e}")
        return None, None


# --- Standard image preprocessing function ---
def preprocess_data(image, label):
    # Simple preprocessing - scale pixel values to [0,1]
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# --- Preprocessing with augmentation ---
def preprocess_and_augment_data(image, label):
    image = data_augmentation(image, training=True) # Apply augmentation
    image = tf.cast(image, tf.float32) / 255.0      # Scale pixel values
    return image, label

# Load datasets
start_time = time.time()
train_ds, class_names = create_dataset(TRAIN_DIR, shuffle=True)
val_ds, _ = create_dataset(VAL_DIR, shuffle=False) # No shuffling for validation
test_ds, _ = create_dataset(TEST_DIR, shuffle=False) # No shuffling for test

if train_ds is None or val_ds is None:
    print("\nError: Failed to load training or validation dataset. Please check paths and directory structure.")
    exit()

# Map preprocessing and optimization
print("Applying preprocessing and optimizing datasets...")
train_ds = train_ds.map(preprocess_and_augment_data, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
if test_ds:
    test_ds = test_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)

# Cache datasets (if they fit in memory, otherwise cache before augmentation/large shuffles)
# Caching after mapping preprocessing is generally good.
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
if test_ds:
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
else:
    # Fallback to validation set if no test set
    print("No separate test directory provided or found. Evaluation will use the validation set.")
    test_ds = val_ds


# Calculate steps (number of batches) - necessary if using cardinality later
# Using experimental_cardinality which returns the number of batches
try:
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    validation_steps = tf.data.experimental.cardinality(val_ds).numpy()
    test_steps = tf.data.experimental.cardinality(test_ds).numpy()
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Test steps: {test_steps}")
    if steps_per_epoch <= 0 or validation_steps <= 0:
         print("\nWarning: Calculated steps per epoch or validation steps is zero or negative.")
         print("Check dataset loading and batch size.")
         # Exit if critical steps are zero
         if steps_per_epoch <= 0: exit()

except tf.errors.OutOfRangeError:
    print("\nWarning: Could not determine dataset cardinality. Steps per epoch/validation might be inaccurate.")
    print("Consider iterating through the dataset once to count elements if needed.")
    steps_per_epoch = None # Let model.fit try to infer
    validation_steps = None
    test_steps = None


print(f"Data pipeline setup time: {time.time() - start_time:.2f} seconds")

# Check class distribution
print("\nChecking class distribution in datasets:")

def count_samples_per_class(dataset):
    class_counts = {0: 0, 1: 0}  # Pre-initialize for binary classification
    # Create a copy of the dataset that's not prefetched or cached
    raw_ds = dataset
    
    # Get total number of batches for progress reporting
    try:
        total_batches = tf.data.experimental.cardinality(raw_ds).numpy()
        if total_batches < 0:  # If unknown, estimate based on typical dataset size
            total_batches = 100
    except:
        total_batches = 100  # Default if we can't determine
    
    print(f"Counting classes across approximately {total_batches} batches...")
    
    # Limit counting to first 20 batches to avoid long processing
    batch_limit = 20
    print(f"Using first {batch_limit} batches for class distribution sampling")
    
    # Use tqdm for progress tracking
    batch_iterator = tqdm_auto(raw_ds.take(batch_limit), total=batch_limit, desc="Counting classes")
    
    # Iterating through batches with tqdm progress
    for _, labels in batch_iterator:
        # Process batch of labels
        label_values = labels.numpy().flatten()
        for label in label_values:
            label_int = int(label)
            class_counts[label_int] += 1
        
        # Update progress bar with current counts
        batch_iterator.set_postfix({
            'real': class_counts[1],
            'fake': class_counts[0]
        })
    
    # Calculate estimated full distribution
    if batch_limit < total_batches:
        scaling_factor = total_batches / batch_limit
        print(f"Sampled {batch_limit} batches. Scaling results by {scaling_factor:.2f} for estimate")
        scaled_counts = {k: int(v * scaling_factor) for k, v in class_counts.items()}
        return scaled_counts
    
    return class_counts

# Count classes in training dataset
try:
    print("Counting classes in training dataset...")
    train_class_counts = count_samples_per_class(train_ds)
    print(f"Class distribution in training set: {train_class_counts}")

    # Count classes in validation dataset
    print("Counting classes in validation dataset...")
    val_class_counts = count_samples_per_class(val_ds)
    print(f"Class distribution in validation set: {val_class_counts}")

    # Check if there's significant class imbalance
    if train_class_counts and len(train_class_counts) == 2:
        keys = list(train_class_counts.keys())
        ratio = max(train_class_counts.values()) / min(train_class_counts.values())
        if ratio > 1.5:
            print(f"\nWarning: Class imbalance detected (ratio: {ratio:.2f})")
            print("Consider using class weights or balancing techniques during training")
            
            # Calculate class weights
            total_samples = sum(train_class_counts.values())
            class_weights = {k: total_samples / (len(train_class_counts) * v) for k, v in train_class_counts.items()}
            print(f"Recommended class weights: {class_weights}")
except Exception as e:
    print(f"Error analyzing class distribution: {e}")
    print("Continuing with model training without class distribution information.")

# --- Build the Parameter-Tuned CNN Model ---
print("\nBuilding the Enhanced CNN Model for 128x128 images...")

from tensorflow.keras.regularizers import l2

# Create the enhanced CNN with regularization to combat overfitting
tuned_cnn = Sequential()

# First Conv block with L2 regularization
tuned_cnn.add(Conv2D(64, (3, 3), padding='same', activation='relu', 
                    kernel_regularizer=l2(0.001),
                    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
tuned_cnn.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)))
tuned_cnn.add(BatchNormalization(momentum=0.95, epsilon=0.005))
tuned_cnn.add(MaxPooling2D((2, 2)))
tuned_cnn.add(Dropout(0.1))  # Light dropout early in the network

# Second Conv block
tuned_cnn.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)))
tuned_cnn.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)))
tuned_cnn.add(BatchNormalization(momentum=0.95, epsilon=0.005))
tuned_cnn.add(MaxPooling2D((2, 2)))
tuned_cnn.add(Dropout(0.2))

# Third Conv block
tuned_cnn.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)))
tuned_cnn.add(BatchNormalization(momentum=0.95, epsilon=0.005))
tuned_cnn.add(MaxPooling2D((2, 2)))
tuned_cnn.add(Dropout(0.3))

# Fourth Conv block - simplified
tuned_cnn.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)))
tuned_cnn.add(BatchNormalization(momentum=0.95, epsilon=0.005))
tuned_cnn.add(MaxPooling2D((2, 2)))
tuned_cnn.add(Dropout(0.4))

# Flatten layer
tuned_cnn.add(Flatten())

# Dense layers - reduced complexity
tuned_cnn.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
tuned_cnn.add(BatchNormalization())
tuned_cnn.add(Dropout(0.5))
tuned_cnn.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
tuned_cnn.add(BatchNormalization())
tuned_cnn.add(Dropout(0.5))

# Output layer
tuned_cnn.add(Dense(1, activation='sigmoid', dtype='float32'))

# Compile the model with Adam optimizer instead of SGD
optimizer = Adam(learning_rate=LEARNING_RATE)
tuned_cnn.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC', 'Precision', 'Recall']  # Track more metrics during training
)

# Display model summary
tuned_cnn.summary()

# Let's create three separate dataset pipelines for each model
# This approach lets us apply the correct preprocessing for each model
print("\nCreating a single optimized dataset pipeline...")

# Let's remove the specialized pipeline sections to simplify our approach
# We'll handle preprocessing differences within the model structure

# --- Train the Parameter-Tuned CNN Model ---
print("Training the enhanced CNN model...")

# Define all necessary callbacks
# Early stopping (already defined earlier)
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=20,
    verbose=1, 
    restore_best_weights=True,
    min_delta=0.001
)

# Add ModelCheckpoint callback
model_checkpoint = ModelCheckpoint(
    'best_highscore_cnn_detector.keras',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# CSV logger
csv_logger = CSVLogger('training_log_highscore_cnn.csv', append=True)

# Create TensorBoard callback for visualization
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    update_freq='epoch'
)

# Create a learning rate scheduler
class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    def __init__(self, lr_max=0.0001, warmup_epochs=5, total_epochs=50):
        super().__init__()
        self.lr_max = lr_max
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.lr_max * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.lr_max * 0.5 * (1 + np.cos(np.pi * progress))
        
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        print(f"\nLearning rate for epoch {epoch+1} set to {lr:.6f}")

# Initialize the learning rate scheduler
lr_scheduler = WarmUpCosineDecayScheduler(
    lr_max=LEARNING_RATE,
    warmup_epochs=3,
    total_epochs=EPOCHS
)

# Update the callback list to only include necessary callbacks
callbacks_list = [early_stopping, model_checkpoint, csv_logger, lr_scheduler, tensorboard_callback]

# Initialize class_weight variable
class_weight = None
if USE_CLASS_WEIGHTS and 'class_weights' in locals():
    print(f"Using class weights: {class_weights}")
    class_weight = class_weights

# Set verbose=1 to ensure per-epoch progress bars are shown alongside the total progress
history = tuned_cnn.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks_list,
    class_weight=class_weight,
    batch_size=32,
    verbose=1  # Changed from 2 to 1 to show per-batch progress
)

print(f"Training finished. Metrics logged to 'training_log_highscore_cnn.csv'.")

# --- Evaluate the Parameter-Tuned CNN Model ---
print("\nEvaluating the Parameter-Tuned CNN Model...")

# Plot training history
def plot_training_history(history):
    """Plot the training and validation accuracy and loss"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('tuned_cnn_training_history.png')
    print("Training history plot saved to 'tuned_cnn_training_history.png'")
    plt.close()  # Close the figure instead of showing it

# Plot the training history
print("Plotting training history...")
plot_training_history(history)

# Load the best model weights
try:
    print("Loading best model weights from checkpoint...")
    # Corrected model checkpoint filename to match what was saved earlier
    tuned_cnn.load_weights('best_highscore_cnn_detector.keras')
except Exception as e:
    print(f"Could not load weights from checkpoint: {e}. Using current model state.")

# Evaluate on the test set
print("\nFinal evaluation on the test set:")
if test_ds is not None:
    # Use tqdm for evaluation progress
    print("Running evaluation with progress bar...")
    
    # Fix for "too many values to unpack" error
    evaluation_results = tuned_cnn.evaluate(test_ds, steps=test_steps, verbose=0)
    loss = evaluation_results[0]
    accuracy = evaluation_results[1]
    auc = evaluation_results[2]
    precision = evaluation_results[3]
    recall = evaluation_results[4]
    
    print(f"Parameter-Tuned CNN Test Loss: {loss:.4f}")
    print(f"Parameter-Tuned CNN Test Accuracy: {accuracy:.4f}")
    print(f"Parameter-Tuned CNN Test AUC: {auc:.4f}")
    print(f"Parameter-Tuned CNN Test Precision: {precision:.4f}")
    print(f"Parameter-Tuned CNN Test Recall: {recall:.4f}")

    # Generate predictions for classification report and confusion matrix
    print("\nGenerating predictions for detailed metrics...")
    predictions_list = []
    labels_list = []
    
    # Create a progress bar for the prediction process
    prediction_progress = tqdm_auto(test_ds, desc="Generating predictions")
    
    for images, labels in prediction_progress:
        batch_preds = tuned_cnn.predict(images, verbose=0)
        predictions_list.append(batch_preds)
        labels_list.append(labels.numpy())
        
        # Update progress bar with current batch stats
        prediction_progress.set_postfix({
            'samples': len(predictions_list) * BATCH_SIZE
        })
    
    # Concatenate results from all batches
    predictions = np.concatenate(predictions_list, axis=0)
    true_classes = np.concatenate(labels_list, axis=0).flatten()
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    
    # Ensure consistent lengths
    min_len = min(len(predicted_classes), len(true_classes))
    predicted_classes = predicted_classes[:min_len]
    true_classes = true_classes[:min_len]
    
    print("\nClassification Report:")
    target_names = ['fake', 'real']
    if class_names:
        target_names = class_names
    print(classification_report(true_classes, predicted_classes, target_names=target_names))
    
    # Generate confusion matrix visualization
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('tuned_cnn_confusion_matrix.png')
    print("Confusion matrix saved to 'tuned_cnn_confusion_matrix.png'")
    plt.close()  # Close the figure instead of showing it
    
    # Calculate additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    precision = precision_score(true_classes, predicted_classes)
    recall = recall_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes)
    try:
        auc = roc_auc_score(true_classes, predictions)
    except:
        auc = 0
        
    print(f"\nDetailed Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

# --- Save the Final Tuned CNN Model ---
print("\nSaving the final trained CNN model...")
try:
    # Standard method with error handling
    tuned_cnn.save('final_tuned_cnn_detector.keras')
    print("Tuned CNN model saved successfully as 'final_tuned_cnn_detector.keras'")
except TypeError as e:
    print(f"Standard save failed with error: {e}")
    print("Trying alternative saving approach...")
    
    # Alternative approach 1: Save only the weights
    tuned_cnn.save_weights('final_tuned_cnn_weights.keras')
    print("Model weights saved as 'final_tuned_cnn_weights.keras'")
    
    # Save the model architecture as JSON
    try:
        model_json = tuned_cnn.to_json()
        with open("tuned_cnn_architecture.json", "w") as json_file:
            json_file.write(model_json)
        print("Model architecture saved as 'tuned_cnn_architecture.json'")
    except Exception as arch_error:
        print(f"Could not save model architecture: {arch_error}")
    
    # Alternative approach 2: Save with TensorFlow's SavedModel format
    try:
        tf.saved_model.save(tuned_cnn, 'tuned_cnn_savedmodel')
        print("Model saved in TensorFlow SavedModel format at 'tuned_cnn_savedmodel'")
    except Exception as tf_error:
        print(f"TensorFlow SavedModel approach failed: {tf_error}")
        
    print("\nTo load this model in the future, use:")
    print("1. Load weights: model.load_weights('final_tuned_cnn_weights.keras')")
    print("   or")
    print("2. Load SavedModel: model = tf.keras.models.load_model('tuned_cnn_savedmodel')")

print("\nExperiment Completed Successfully!")
print("-----------------------------------------------")
print(f"Dataset: {DATASET_DIR}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Batch Size: {BATCH_SIZE}")  # Updated to just show the current batch size
print(f"Training Epochs: {EPOCHS}")
print(f"Final Metrics:")
print(f"  - Accuracy: {accuracy:.4f}")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall: {recall:.4f}")
print(f"  - F1 Score: {f1:.4f}")
print(f"  - AUC: {auc:.4f}")
print("-----------------------------------------------")

# Save final summary to a separate file
with open('final_scores_summary.txt', 'w') as f:
    f.write("Deepfake Detection Final Results\n")
    f.write("===============================\n")
    f.write(f"Dataset: {DATASET_DIR}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Training Epochs: {EPOCHS}\n\n")
    f.write("Final Metrics:\n")
    f.write(f"  - Accuracy: {accuracy:.4f}\n")
    f.write(f"  - Precision: {precision:.4f}\n")
    f.write(f"  - Recall: {recall:.4f}\n")
    f.write(f"  - F1 Score: {f1:.4f}\n")
    f.write(f"  - AUC: {auc:.4f}\n")
    f.write("===============================\n")

print("Final summary saved to 'final_scores_summary.txt'")