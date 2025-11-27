from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1️⃣ Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen   = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224,224),     # ResNet50 expects 224x224
    batch_size=32,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

# 2️⃣ Load ResNet50 without top classifier
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# 3️⃣ Add custom classifier head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4️⃣ Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5️⃣ Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# 6️⃣ Save model
model.save("resnet50_model.h5")