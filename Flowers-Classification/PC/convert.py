import tensorflow as tf

# Step 1: Load the Keras H5 model
model = tf.keras.models.load_model('/home/mjayakumar/VVDN_BU_PROJECTS/tensorflowlite/Flowers Recognition/flowersOption2.h5')

# Step 2: Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Step 3: Save the TFLite model
with open('/home/mjayakumar/VVDN_BU_PROJECTS/tensorflowlite/Flowers Recognition/flowersOption2.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Converted successfully to TFLite format!")

