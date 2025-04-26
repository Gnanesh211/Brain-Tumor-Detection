# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adamax
from PIL import Image
import numpy as np
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend access

# Load the model
MODEL_PATH = 'model/Brain Tumors Classifier.h5'
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def preprocess_image(image, target_size=(224, 224)):
    """Resize and normalize the image for model prediction."""
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image /= 255.0  # Normalize if required by the model
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file:
        return jsonify({"error": "Invalid file"}), 400

    # Open and preprocess the image
    image = Image.open(io.BytesIO(file.read()))
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

# Make predictions
    predictions = model.predict(img_array)

    # Predict and print confidence scores for debugging
    # print(predictions)
    confidence_scores = predictions[0].tolist()
    print("Debug - Confidence Scores:", confidence_scores)  # Debugging line

    # Determine the predicted class
    class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Print the predicted class
    predicted_class = class_labels[np.argmax(predictions)]
    print("Debug - Predicted Class:", predicted_class)  # Debugging line

    response = {
        "confidence_scores": {class_labels[i]: score for i, score in enumerate(confidence_scores)},
        "predicted_class": predicted_class
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)