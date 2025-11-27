import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Basic settings
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10  # you can increase to 15–20 if it trains fast

train_dir = "dataset/train"
val_dir = "dataset/val"

# 2. Data generators (with a bit of augmentation)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"  # 0/1 labels
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

print("Class indices:", train_gen.class_indices)
# Example: {'fake': 0, 'real': 1} or vice versa

# 3. Simple CNN model (lightweight)
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # output = probability of class "1"
])

model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=1e-4),
    metrics=["accuracy"]
)

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# 4. Save model
os.makedirs("saved_models", exist_ok=True)
model.save("saved_models/deepfake_model.h5")
print("Model saved to saved_models/deepfake_model.h5")
