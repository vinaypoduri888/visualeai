# Install necessary packages


# Import necessary libraries
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import numpy as np
import cv2

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Define the Streamlit application
def main():
    st.title("Visual Question Answering with BLIP")

    # Add a sidebar with information
    st.sidebar.title("About This App")
    st.sidebar.write("## Features")
    st.sidebar.write("""
    - Upload an image
    - Ask a question about the image
    - Get an AI-generated answer
    """)
    st.sidebar.write("## How to Use")
    st.sidebar.write("""
    1. Upload an image using the uploader.
    2. Enter a question about the image in the text input.
    3. View the generated answer below the input.
    """)
    st.sidebar.write("### Developed by Vinay")


    # Capture or upload an image
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if image_file is not None:
        raw_image = Image.open(image_file).convert('RGB')
        st.image(raw_image, caption='Uploaded Image', use_column_width=True)

        # Ask a question
        question = st.text_input("Ask a question about the image:")

        if question:
            # Process the inputs
            inputs = processor(raw_image, question, return_tensors="pt")

            # Generate the output
            out = model.generate(**inputs)

            # Decode and print the result
            answer = processor.decode(out[0], skip_special_tokens=True)
            st.write("Answer: ", answer)

# Run the Streamlit application
if __name__ == "__main__":
    main()
