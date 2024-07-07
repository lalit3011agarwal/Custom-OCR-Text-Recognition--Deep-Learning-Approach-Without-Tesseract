from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import StringLookup

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the CTC loss function
def ctc_loss(y_true, y_pred, input_length, label_length):
    y_true = tf.cast(y_true, tf.int32)
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=label_length,
        logit_length=input_length,
        blank_index=len(chars),
    )
    return tf.reduce_mean(loss)

# Load the trained model and character mappings
model = keras.models.load_model('my_model1.keras', custom_objects={'ctc_loss': ctc_loss})
chars = list(['!', '"', '#', '$', '%', '&', "'", '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', '^', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
)  # replace with your actual character set
char_to_num = StringLookup(vocabulary=list(chars), mask_token=None)
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def preprocess_image(image_path):
    """Loads and preprocesses an image for model prediction."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image could not be loaded.")
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if 100 < area < 5000 and (0.2 < w / h < 10):
            bounding_boxes.append([x, y, w, h])
    processed_images = []
    for bbox in bounding_boxes:
        x_min, y_min, width, height = map(int, bbox)
        cropped_img = image[y_min:y_min + height, x_min:x_min + width]
        height = 32
        aspect_ratio = cropped_img.shape[1] / cropped_img.shape[0]
        width = min(int(height * aspect_ratio), 128)
        cropped_img = cv2.resize(cropped_img, (width, height)) / 255.0
        padded_img = np.zeros((32, 128))
        padded_img[:, :width] = cropped_img
        processed_images.append(padded_img)
    return processed_images, bounding_boxes

def predict_text(image_path, model, char_to_num, num_to_char):
    processed_images, _ = preprocess_image(image_path)
    predicted_texts = []
    for img in processed_images:
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        decoded_prediction = tf.keras.backend.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1], greedy=True)[0][0]
        predicted_text = tf.strings.reduce_join(num_to_char(decoded_prediction)).numpy().decode("utf-8")
        predicted_texts.append(predicted_text)
    return predicted_texts

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            extracted_text = predict_text(file_path, model, char_to_num, num_to_char)
            return render_template('result.html', text=extracted_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
