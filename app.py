import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model_url = 'https://github.com/shahidbnriaz/potato_leaf_prediction/raw/main/model.keras'
    try:
        response = requests.get(model_url)
        response.raise_for_status()  # Raise an exception for 4xx/5xx errors
        model_file = BytesIO(response.content)
        model = tf.keras.models.load_model(model_file)
        return model
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching model from GitHub: {e}")
        return None

# Load the model
model = load_model()

if model is not None:
    # Now you can use the model in your Streamlit app
    # Example:
    # prediction = model.predict(input_data)
    st.write("Model loaded successfully!")
else:
    st.write("Failed to load the model. Check the logs for details.")

# Define class names
class_names = ['Class1', 'Class2', 'Class3']  # Replace with your actual class names

def predict(image):
    image = np.array(image)
    image = tf.image.resize(image, (224, 224))  # Adjust size as per your model's requirement
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# Streamlit app
st.title("Potato Disease Prediction")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_class, confidence = predict(image)
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
