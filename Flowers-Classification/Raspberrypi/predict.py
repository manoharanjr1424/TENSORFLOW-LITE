import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="/home/mano/Ml/flower_classification/flowersOption2.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Load your image (ensure it's RGB)
image_path = "/home/mano/Ml/flower_classification/original.jpg"  # Replace with your test image
img = Image.open(image_path).convert("RGB").resize((64, 64))
img = np.array(img, dtype=np.float32) / 255.0  # Normalize like training
img = np.expand_dims(img, axis=0)  # Shape: (1, 64, 64, 3)

# Run inference
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

# Get prediction
output = interpreter.get_tensor(output_details[0]['index'])
predicted_index = np.argmax(output)

# ⚠️ Make sure the order matches `train_set.class_indices`
class_labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
print("Predicted flower:", class_labels[predicted_index])
