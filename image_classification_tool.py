import tensorflow as tf
from tensorflow import keras
from tkinter import Tk, filedialog, Label, Button, Canvas, PhotoImage
from PIL import Image, ImageTk
import numpy as np

# Load a pre-trained CNN model (replace with your own model if needed)
model = keras.applications.MobileNetV2(weights='imagenet')

def classify_image(image_path):
    """
    Classify the input image using the pre-trained model.
    """
    img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = keras.applications.mobilenet_v2.decode_predictions(predictions.numpy())[0]

    return decoded_predictions

def open_file_dialog():
    """
    Open a file dialog to choose an image file and classify it.
    """
    file_path = filedialog.askopenfilename()
    if file_path:
        classify_and_display_image(file_path)

def classify_and_display_image(image_path):
    """
    Classify the image and display the result on the GUI.
    """
    predictions = classify_image(image_path)

    # Display the result in the Tkinter window
    result_text.set(f"Prediction:\n{predictions[0][1]} ({predictions[0][2]*100:.2f}%)")

    # Display the image in the Tkinter window
    img = Image.open(image_path)
    img = img.resize((300, 300))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

# Create the Tkinter window
root = Tk()
root.title("Image Classification Tool")

# Create Tkinter variables
result_text = Label(root, text="", font=("Helvetica", 14))
result_text.pack(pady=20)

# Create a label to display the image
image_label = Label(root)
image_label.pack()

# Create a button to open the file dialog
classify_button = Button(root, text="Classify Image", command=open_file_dialog)
classify_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
