import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Load the trained model
model = keras.models.load_model("my_model.keras")

# Path to the test image (update this)
img_path = "test_images/example.jpg"  # ← change this to your test image path

# Load and preprocess the image
img = image.load_img(img_path, target_size=(128, 128))  # same size used during training
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)

# Predict
prediction = model.predict(img_array)

# Interpret the result
if prediction[0][0] > 0.5:
    print("✅ Predicted class: Class 1")
else:
    print("✅ Predicted class: Class 0")

print("🔢 Raw prediction score:", prediction[0][0])

