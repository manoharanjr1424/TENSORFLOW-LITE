import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Load model
model = load_model("/home/mjayakumar/VVDN_BU_PROJECTS/tensorflowlite/Flowers Recognition/flowersOption2.h5")

# Load and preprocess image
img_path = "/home/mjayakumar/VVDN_BU_PROJECTS/tensorflowlite/Flowers Recognition/sunflower.jpg"
img = load_img(img_path, target_size=(64, 64))  # Match with model's input shape
img_array = img_to_array(img)
img_array = img_array / 255.0                   # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

# Optional: class names
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
print(f"Prediction: {class_names[predicted_class]} ({confidence:.2f}%)")
