from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = load_model("model.h5")

# Path to a single image you want to test
img_path = r"C:\fyyp\dataset\val\fake\fake.png"

# Preprocess the image
img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)[0][0]

# Print the output
print("Prediction value:", pred)
if pred > 0.5:
    print("Detected as Deepfake")
else:
    print("Detected as Real")