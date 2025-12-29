"""
Training script for Paddy Crop Disease Classification Model
Uses EfficientNetB0 as base model with transfer learning and data augmentation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# Set image dimensions and batch size
IMG_SIZE = 224
BATCH_SIZE = 32  # Reduce if you get memory errors
EPOCHS = 50 # Increased epochs for better learning
NUM_CLASSES = 0  # Will be determined from dataset

# Path to dataset
DATA_DIR = 'cleaned_dataset/tomato'

print("=" * 60)
print("TOMATO CROP DISEASE CLASSIFICATION MODEL TRAINING")
print("=" * 60)

# Get number of classes from directory structure
if os.path.exists(DATA_DIR):
    NUM_CLASSES = len([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"\nFound {NUM_CLASSES} disease classes in {DATA_DIR}")
    
    # List all classes
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    print("Classes found:", classes)
else:
    print(f"Error: Dataset directory {DATA_DIR} not found!")
    exit(1)

# Create ImageDataGenerator with data augmentation for training
# Data augmentation helps improve model generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,                    # Normalize pixel values to [0, 1]
    rotation_range=20,                 # Random rotation up to 20 degrees
    width_shift_range=0.2,            # Random horizontal shift
    height_shift_range=0.2,           # Random vertical shift
    shear_range=0.2,                  # Random shear transformation
    zoom_range=0.2,                   # Random zoom
    horizontal_flip=True,             # Random horizontal flip
    fill_mode='nearest',              # Fill mode for transformations
    validation_split=0.2              # 20% for validation
)

# Validation generator (only rescaling, no augmentation)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation data generator
val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Build the model using EfficientNetB0 as base
print("\nBuilding model with EfficientNetB0 base...")

# Try to load pre-trained model - MobileNetV2 often works better with Keras 3.x
# If EfficientNet fails, we'll use MobileNetV2 as fallback
print("Loading pre-trained base model...")
base_model = None
use_mobilenet = False

# First try EfficientNetB0
try:
    print("Attempting to load EfficientNetB0 with ImageNet weights...")
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg'  # Add pooling directly
    )
    print("✓ Successfully loaded EfficientNetB0 with ImageNet weights!")
except (ValueError, Exception) as e:
    print(f"⚠ EfficientNetB0 failed: {str(e)[:150]}")
    print("Trying MobileNetV2 (more compatible with Keras 3.x)...")
    try:
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            pooling='avg'
        )
        use_mobilenet = True
        print("✓ Successfully loaded MobileNetV2 with ImageNet weights!")
    except Exception as e2:
        print(f"⚠ MobileNetV2 also failed: {str(e2)[:150]}")
        print("Loading MobileNetV2 without pre-trained weights...")
        base_model = MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            pooling='avg'
        )
        use_mobilenet = True
        print("⚠ Warning: Training from scratch (no pre-trained weights)")
        print("⚠ This will require more epochs to achieve good accuracy!")

# Freeze base model layers initially (we'll unfreeze later for fine-tuning)
base_model.trainable = False

# Add custom classification head with improved architecture
# The base model already has pooling='avg', so output is already flattened
x = base_model.output
# If needed, add GlobalAveragePooling2D (but pooling='avg' in base_model should handle this)
# Check if we need additional pooling
if len(x.shape) > 2:
    x = GlobalAveragePooling2D()(x)

# Improved dense layers with batch normalization
x = Dense(1024, activation='relu')(x)       # Larger dense layer for better capacity
x = Dropout(0.4)(x)                         # Slightly reduced dropout
x = Dense(512, activation='relu')(x)       # Additional dense layer
x = Dropout(0.3)(x)                         # Lower dropout in second layer
predictions = Dense(NUM_CLASSES, activation='softmax')(x)  # Output layer

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with improved settings
# Use a lower initial learning rate for more stable training
initial_lr = 0.0005 if use_mobilenet else 0.001
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=initial_lr, beta_1=0.9, beta_2=0.999),
    loss='categorical_crossentropy',
    metrics=['accuracy']  # Monitor accuracy
)

print("Model architecture:")
model.summary()

# Define callbacks for training
callbacks = [
    # Save the best model during training
    ModelCheckpoint(
        'model/tomato_disease_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Stop training if validation accuracy doesn't improve (increased patience)
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,  # Increased patience for better convergence
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001  # Minimum change to qualify as improvement
    ),
    # Reduce learning rate if validation loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]

# Train the model
print("\nStarting training...")
print("-" * 60)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Fine-tuning: Unfreeze base model layers gradually
print("\nFine-tuning model...")
print("Unfreezing last few layers first, then all layers...")

# Strategy 1: Unfreeze only last few layers first (for EfficientNet/MobileNet)
if not use_mobilenet:
    # For EfficientNet, unfreeze last 30 layers first
    for layer in base_model.layers[-30:]:
        layer.trainable = True
else:
    # For MobileNetV2, unfreeze last 20 layers first
    for layer in base_model.layers[-20:]:
        layer.trainable = True

print(f"Unfroze last layers. Training with lower learning rate...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune with partial unfreezing
history_partial = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=15,  # More epochs for partial fine-tuning
    callbacks=callbacks
)

# Strategy 2: Now unfreeze all layers for final fine-tuning
print("\nUnfreezing all layers for final fine-tuning...")
base_model.trainable = True

# Even lower learning rate for full fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Final fine-tuning with all layers
history_finetune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=15,  # Final fine-tuning epochs
    callbacks=callbacks
)

# Save the final model
model.save('model/tomato_disease_model.h5')
print("\n" + "=" * 60)
print("Training completed! Model saved to: model/tomato_disease_model.h5")
print("=" * 60)
   