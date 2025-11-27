from flask import Flask, render_template, request, jsonify, session, url_for, redirect
import gspread
import os
import gdown
from google.oauth2.service_account import Credentials
from datetime import datetime
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

from dotenv import load_dotenv

load_dotenv()
# ==============================
# 🔥 LOAD XCEPTION DEEPFAKE MODEL
# ==============================
MODEL_PATH = os.path.join("saved_models/xception_deepfake.h5")


def download_model():
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    if not os.path.exists(MODEL_PATH):
        print("Model not found. Downloading...")
        url = "https://github.com/aqeefrusydan-hub/fyp-faceguardAI/releases/download/v1.0/xception_deepfake.h5"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model download complete!")


# Call before loading model
download_model()

model = load_model(MODEL_PATH)
print("✅ Xception Deepfake model loaded successfully")

# ==============================
# Flask Setup
# ==============================
app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Required for session login tracking


# ==============================
# GOOGLE SHEETS SETUP
# ==============================


def get_sheet():
    # Load JSON from environment variable
    creds_json = os.getenv("GOOGLE_CREDS")
    creds_dict = json.loads(creds_json)

    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )

    client = gspread.authorize(creds)
    sheet = client.open("Database").sheet1
    return sheet


# ==============================
# IMAGE PREDICTION FUNCTION
# ==============================
IMG_SIZE = 299  # Xception required size


def predict_image(img_path):
    # Load image and resize
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # 🔥 Xception-specific preprocessing
    img_array = preprocess_input(img_array)

    # Predict
    output = model.predict(img_array)[0][0]

    prob_real = float(output)
    prob_fake = 1 - prob_real

    label = "FAKE" if prob_fake >= 0.5 else "REAL"

    return {
        "label": label,
        "fake_percent": round(prob_fake * 100, 2),
        "real_percent": round(prob_real * 100, 2)
    }


# ==============================
# ROUTES
# ==============================
@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/extension')
def extension():
    return render_template('index.html')


@app.route("/logout")
def logout():
    session.clear()  # clears login session
    return redirect("/")  # send user back to homepage


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        email = data["email"]
        password = data["password"]

        sheet = get_sheet()
        users = sheet.get_all_records()

        for user in users:
            if user["email"] == email and user["password"] == password:
                session["logged_in"] = True
                session["user_email"] = email
                return jsonify({"message": "Login successful"}), 200

        return jsonify({"error": "Invalid email or password"}), 400

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json()
        fullname = data["fullname"]
        email = data["email"]
        password = data["password"]

        sheet = get_sheet()
        sheet.append_row([fullname, email, password, str(datetime.now())])

        return jsonify({"message": "Signup successful"}), 200

    return render_template('signup.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    pred = predict_image(filepath)

    return render_template(
        'result.html',
        label=pred["label"],
        fake_percent=pred["fake_percent"],
        real_percent=pred["real_percent"],
        img_path=filepath
    )


# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
