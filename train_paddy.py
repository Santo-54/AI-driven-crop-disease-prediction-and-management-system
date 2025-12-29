import tensorflow as tf
from tensorflow import keras
# using V2 which is more stable in Keras 3
from tensorflow.keras.applications import EfficientNetV2B0 
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# Clear session
tf.keras.backend.clear_session()

# --- TOMATO CONFIGURATION ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
TRAIN_DIR = 'data/paddy/train'
VAL_DIR = 'data/paddy/validation'

print("="*60)
print("STARTING TRAINING: PADDY MODEL (EfficientNetV2)")
print("Target File: model/paddy_disease_model.h5")
print("="*60)

# Check Directories
if os.path.exists(TRAIN_DIR):
    classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    NUM_CLASSES = len(classes)
    print(f"Classes found: {classes}")
else:
    print("Error: Train directory not found")
    exit(1)

# --- DATA GENERATORS ---
# IMPORTANT: EfficientNetV2 expects pixels 0-255. 
# WE REMOVED 'rescale=1./255' intentionally. DO NOT ADD IT BACK.

train_datagen = ImageDataGenerator(
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2,
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest'
    # No rescale here!
)

val_datagen = ImageDataGenerator(
    # No rescale here!
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

# --- MODEL ARCHITECTURE (EfficientNetV2B0) ---
print("Building EfficientNetV2B0...")

# EfficientNetV2B0 handles the input shape much better in Keras 3
base_model = EfficientNetV2B0(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_preprocessing=True # This handles the scaling internally
)
base_model.trainable = False

x = base_model.output
# Note: V2B0 usually includes pooling, but we add it to be safe
if len(x.shape) > 2:
    x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    ModelCheckpoint('model/paddy_disease_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
]

# Training
print("Starting training...")
model.fit(train_generator, steps_per_epoch=train_generator.samples//BATCH_SIZE,
          validation_data=val_generator, validation_steps=val_generator.samples//BATCH_SIZE,
          epochs=EPOCHS, callbacks=callbacks)

# Fine-Tuning
print("Unfreezing for fine-tuning...")
base_model.trainable = True
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, steps_per_epoch=train_generator.samples//BATCH_SIZE,
          validation_data=val_generator, validation_steps=val_generator.samples//BATCH_SIZE,
          epochs=15, callbacks=callbacks)

model.save('model/paddy_disease_model.h5')
print("SUCCESS: Saved model/paddy_disease_model.h5")