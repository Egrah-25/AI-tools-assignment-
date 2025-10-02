# app.py (Streamlit deployment for MNIST classifier)
"""
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_cnn_model.h5')

def preprocess_image(image):
    # Convert to grayscale and resize
    image = image.convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

def main():
    st.title("MNIST Handwritten Digit Classifier")
    st.write("Upload an image of a handwritten digit (0-9) for classification")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess and predict
        processed_image = preprocess_image(image)
        
        # Load model and predict
        model = load_model()
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Display results
        st.write(f"**Predicted Digit:** {predicted_digit}")
        st.write(f"**Confidence:** {confidence:.2%}")
        
        # Show probability distribution
        fig, ax = plt.subplots()
        ax.bar(range(10), prediction[0])
        ax.set_xlabel('Digit')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
"""
