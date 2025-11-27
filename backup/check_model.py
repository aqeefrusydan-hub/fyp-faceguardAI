from tensorflow.keras.models import load_model

model = load_model("saved_models/deepfake_detector.h5")
model.summary()
