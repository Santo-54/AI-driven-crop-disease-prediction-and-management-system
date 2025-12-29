"""
Training script for Chilly Crop Disease Classification Model
Uses EfficientNetB0 as base model with transfer learning and data augmentation
"""

import tensorflow as tf
from tensorflow import keras
from keras.applications import EfficientNetB0
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# Set image dimensions and batch size
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 0  # Will be determined from dataset

# Path to dataset
DATA_DIR = 'dataset/chilly'

print("=" * 60)
print("CHILLY CROP DISEASE CLASSIFICATION MODEL TRAINING")
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

# Check if dataset is empty
if NUM_CLASSES == 0:
    print(f"\nWarning: No disease classes found in {DATA_DIR}!")
    print("Please ensure your chilly dataset is organized in folders.")
    print("Each folder should represent a disease class.")
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

# Load EfficientNetB0 pre-trained on ImageNet (excluding top layers)
# Note: Keras 3.x may have compatibility issues with ImageNet weights
print("Loading EfficientNetB0 base model...")
try:
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    print("✓ Loaded EfficientNetB0 with ImageNet weights")
except (ValueError, Exception) as e:
    print(f"⚠ Warning: Could not load ImageNet weights ({str(e)[:100]})")
    print("Loading model without pre-trained weights (will train from scratch)...")
    base_model = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    print("✓ Loaded EfficientNetB0 without pre-trained weights")

# Freeze base model layers initially (we'll unfreeze later for fine-tuning)
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)      # Global average pooling to reduce dimensions
x = Dense(512, activation='relu')(x)  # Dense layer with ReLU activation
x = Dropout(0.5)(x)                   # Dropout for regularization
predictions = Dense(NUM_CLASSES, activation='softmax')(x)  # Output layer

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model architecture:")
model.summary()

# Define callbacks for training
callbacks = [
    # Save the best model during training
    ModelCheckpoint(
        'model/chilly_disease_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Stop training if validation accuracy doesn't improve
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
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

# Fine-tuning: Unfreeze base model layers and train with lower learning rate
print("\nFine-tuning model...")
base_model.trainable = True

# Use lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train again with unfrozen layers
history_finetune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=10,  # Fewer epochs for fine-tuning
    callbacks=callbacks
)

# Save the final model
model.save('model/chilly_disease_model.h5')
print("\n" + "=" * 60)
print("Training completed! Model saved to: model/chilly_disease_model.h5")
print("=" * 60)
