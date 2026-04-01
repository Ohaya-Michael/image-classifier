import tensorflow as tf
import numpy as np
from .preprocessing import preprocess_image
import os


# Define your custom logic
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def predict_image(model_path, img):
    """
    Predict the class of the image using the provided model.
    Args:
        model_path (str): Path to the trained TensorFlow model.
        img (PIL.Image): The image to predict.
    Returns:
        dict: A dictionary containing the top 5 predicted classes and their probabilities.
    """
    # Optional: Define class names if you have them
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    # 1. Bild vorverarbeiten
    preprocessed_image = preprocess_image(img)
    top_5_predictions = {}
  
    # Load the model and provide the mapping
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'custom_loss': custom_loss}
    )
    # Vorhersage ausführen
    predictions = model.predict(preprocessed_image)[0] # [0], um aus der Batch-Dimension (1, N) zu (N,) zu kommen

    # Die Top-5 Indizes finden (sortiert nach Wahrscheinlichkeit)
    top_5_indices = np.argsort(predictions)[-5:][::-1]

    for i in top_5_indices:
        score = predictions[i]
        # Falls class_names existiert: label = class_names[i]
        # Sonst nutzen wir den Index:
        label = class_names[i] if i < len(class_names) else f"Class {i}"
        top_5_predictions[label] = f"{score:.2%}"
        
    return top_5_predictions


# top_5_predictions = predict_image("models/trained_model_finetuned.keras", '../uploads/test_image.jpg')
# print(top_5_predictions)