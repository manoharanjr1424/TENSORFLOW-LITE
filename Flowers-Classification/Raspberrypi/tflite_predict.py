import numpy as np
import cv2
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="/home/mano/Ml/flower_classification/flowersOption2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors info
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
img = cv2.imread("/home/mano/Ml/flower_classification/sunflower.jpg")  # Your test image
img_resized = cv2.resize(img, (64, 64))  # Use 64x64 if model trained on that
img_normalized = img_resized.astype(np.float32) / 255.0
img_input = np.expand_dims(img_normalized, axis=0)  # shape: (1, 64, 64, 3)

# Set the input
interpreter.set_tensor(input_details[0]['index'], img_input)

# Run inference
interpreter.invoke()

# Get prediction
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

# Get label index
predicted_index = np.argmax(output_data)

# Flower labels (sorted like in training)
labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
predicted_label = labels[predicted_index]

print("🌸 Predicted flower:", predicted_label)

