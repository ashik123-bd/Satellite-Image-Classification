from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
import tensorflow as tf
import os

app = Flask(__name__)

# Categories for your dataset
categories = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
              'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Load your trained model
model = tf.keras.models.load_model('model.h5') # Update path to your model file

# Preprocess images for prediction
def preprocess_images(img_paths):
    images = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = img.reshape(128, 128, 3)
        images.append(img)
    images = np.array(images).reshape(len(images), 1, 128, 128, 3)
    return images

# Function to predict categories
def predict_images(img_paths):
    preprocessed_imgs = preprocess_images(img_paths)
    predictions = model.predict(preprocessed_imgs)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_categories = [categories[pred] for pred in predicted_classes]
    confidence_scores = [float(predictions[i][pred]) * 100 for i, pred in enumerate(predicted_classes)]
    return predicted_categories, confidence_scores

# Route to handle image upload and prediction
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Please upload a file"

        file = request.files['file']
        if file.filename == '':
            return "Please select a file"

        # Ensure the uploads directory exists
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        # Save the uploaded image
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Predict category and confidence level
        predicted_category, confidence_score = predict_images([file_path])
        
        return jsonify({
            'filename': file.filename,
            'prediction': predicted_category[0],
            'confidence': f"{confidence_score[0]:.2f}%"
        })

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
