import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import download_model  # Assurez-vous que ce module est dans le même répertoire

# Télécharger le modèle si ce n'est pas déjà fait
model_path = 'potato_leaf_model.h5'
if not os.path.exists(model_path):
    download_model.download_file_from_google_drive('https://drive.google.com/uc?export=download&id=1PUKiY2Q216ZMdKx8LujRehs9YDFwVXoy', model_path)

# Charger le modèle
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict(image):
    if model is None:
        return "Model not loaded"
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

# Créer une interface Gradio
demo = gr.Interface(fn=predict, inputs="image", outputs="label", title="Potato Leaf Disease Detection")
demo.launch()
