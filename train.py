import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_mini_xception
from preprocess import train_generator, val_generator

# Build model
model = build_mini_xception()

# Set up early stopping and model checkpoint
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('mini_xception_best.keras', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save('mini_xception_final.keras')