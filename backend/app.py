from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adamax
from PIL import Image
import numpy as np
import os
import gdown
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend access

# Define the model path
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "Brain Tumors Classifier.h5")
MODEL_URL = "https://drive.google.com/uc?id=1XKhMuY7avZ_Gjq39_RT6P7TMNu3GKhh_"

# Ensure the model directory exists and download the model if missing
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if not os.path.exists(MODEL_PATH):
    print(f"Downloading model to {MODEL_PATH}...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Model downloaded successfully!")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# Compile only if needed for inference (optional)
# model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def preprocess_image(image, target_size=(224, 224)):
    """Resize and normalize the image for model prediction."""
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image /= 255.0  # Normalize to [0, 1]
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file:
        return jsonify({"error": "Invalid file"}), 400

    # Open and preprocess the image using the defined function
    image = Image.open(io.BytesIO(file.read()))
    img_array = preprocess_image(image)

    # Make predictions
    predictions = model.predict(img_array)
    confidence_scores = predictions[0].tolist()
    print("Debug - Confidence Scores:", confidence_scores)  # Debugging line

    # Determine the predicted class
    predicted_class = class_labels[np.argmax(predictions)]

    print("Debug - Predicted Class:", predicted_class)  # Debugging line

    response = {
        "confidence_scores": {class_labels[i]: float(score) for i, score in enumerate(confidence_scores)},
        "predicted_class": predicted_class
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)