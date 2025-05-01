import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Parameters
img_size = (48, 48)
batch_size = 64

# Directories
train_dir = "dataset/train"
test_dir = "dataset/test"

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1  # Keep a validation split
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

if not os.path.exists('preprocessed'):
    os.makedirs('preprocessed')
    
# Save class indices for later mapping
np.save("preprocessed/class_indices.npy", train_generator.class_indices)

print("✅ Data generators created.")
