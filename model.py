import tensorflow as tf
from keras import layers, models

# Mini-Xception model
def build_mini_xception(input_shape=(48, 48, 1), num_classes=7):
    model = models.Sequential()

    # Initial Conv block
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Depthwise separable convolution block 1
    model.add(layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Depthwise separable convolution block 2
    model.add(layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Depthwise separable convolution block 3
    model.add(layers.SeparableConv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout to avoid overfitting
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create and print summary of the model
model = build_mini_xception()
model.summary()
