import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Define class labels (same order as training class_indices)
class_labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="flowersOption2.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Predict flower from image
def predict_flower(image_path):
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output)
    return class_labels[predicted_index]

# GUI Application
class FlowerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flower Classifier Machine Leraning ")
        self.root.geometry("400x450")
        self.root.configure(bg="#F0F0F0")

        self.label = Label(root, text="Upload an image", font=("Arial", 14), bg="#F0F0F0")
        self.label.pack(pady=10)

        self.image_label = Label(root)
        self.image_label.pack()

        self.result_label = Label(root, text="", font=("Arial", 16, "bold"), fg="green", bg="#F0F0F0")
        self.result_label.pack(pady=20)

        self.upload_button = Button(root, text="Choose Image", command=self.upload_image, font=("Arial", 12))
        self.upload_button.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            img = Image.open(file_path).resize((200, 200))
            tk_img = ImageTk.PhotoImage(img)
            self.image_label.configure(image=tk_img)
            self.image_label.image = tk_img

            # Predict and show result
            predicted_flower = predict_flower(file_path)
            self.result_label.configure(text=f"Predicted: {predicted_flower}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FlowerApp(root)
    root.mainloop()

