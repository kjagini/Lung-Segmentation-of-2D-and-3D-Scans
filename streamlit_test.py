#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import cv2
from PIL import Image
import numpy as np

def app():
    st.title("Image Segmentation")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Load image
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Image", use_column_width=True)

        # Perform image segmentation
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # perform image segmentation using OpenCV
        # ...
        
        # Display the segmented image
        segmented_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        st.image(segmented_image, caption="Segmented Image", use_column_width=True)

if __name__ == '__main__':
    app()


# In[ ]:




