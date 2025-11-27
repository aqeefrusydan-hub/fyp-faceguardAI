from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

app = Flask(__name__)

# Load Keras pretrained ResNet50 (ImageNet weights)
model = ResNet50(weights="imagenet")
print("✅ ResNet50 model loaded successfully!")

def predict_image(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=1)[0][0]  # top-1 prediction
    class_name = decoded[1]  # human-readable class
    confidence = decoded[2]  # confidence score
    return f"{class_name} ({confidence*100:.2f}%)"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Save uploaded file
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    result = predict_image(filepath)
    return render_template('result.html', result=result, img_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
