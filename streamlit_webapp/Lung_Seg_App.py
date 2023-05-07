import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import numpy as np
import matplotlib.pyplot as plt

# Load the model
from keras.layers import ConvLSTM2D, Bidirectional, BatchNormalization, Conv3D, Cropping3D, ZeroPadding3D
from conv import ConvRNN2D, ConvFF2DCell, ConvFF2D
from keras.models import Sequential

def get_the_prediction(loaded_data):    
    sim_model = Sequential()
    sim_model.add(BatchNormalization(input_shape = (None, None, None, 1)))
    sim_model.add(Conv3D(8, 
                         kernel_size = (1, 5, 5), 
                         padding = 'same',
                         activation = 'relu'))
    sim_model.add(Conv3D(8, 
                         kernel_size = (3, 3, 3), 
                         padding = 'same',
                         activation = 'relu'))
    sim_model.add(BatchNormalization())
    sim_model.add(Bidirectional(ConvFF2D(16, 
                                           kernel_size = (3, 3),
                                           padding = 'same',
                                           return_sequences = True)))
    sim_model.add(Bidirectional(ConvFF2D(32, 
                                           kernel_size = (3, 3),
                                           padding = 'same',
                                           return_sequences = True)))
    sim_model.add(Conv3D(8, 
                         kernel_size = (1, 3, 3), 
                         padding = 'same',
                         activation = 'relu'))
    sim_model.add(Conv3D(1, 
                         kernel_size = (1,1,1), 
                         activation = 'sigmoid'))
    sim_model.add(Cropping3D((1, 2, 2))) # avoid skewing boundaries
    sim_model.add(ZeroPadding3D((1, 2, 2)))

    sim_model.load_weights('convlstm_model_weights.best.hdf5')

    pred_seg = sim_model.predict(loaded_data)

    return pred_seg

# Streamlit app
st.title("Lung Segmentation")

# Upload the npy file
uploaded_file = st.file_uploader("Choose an npy file...", type="npy")

if uploaded_file is not None:
    # Read the npy file and display it
    loaded_data = np.load(uploaded_file)
    
    st.write("Original Image and Predicted Image")
    # Create a box around the images
    box_style = "border:2px solid #4CAF50;border-radius:5px;padding:10px;margin-bottom:20px;"
    st.markdown(f'<div style="{box_style}">', unsafe_allow_html=True)

    # Display the original image and predicted image side by side
    col1, col2 = st.beta_columns(2)
    with col1:
        st.write("Original Image")
        plt.figure(figsize=(8, 8))
        plt.imshow(loaded_data[0][100])
        plt.axis('off')
        st.pyplot()
    with col2:
        st.write("Predicted Image")
        plt.figure(figsize=(8, 8))
        plt.imshow(get_the_prediction((loaded_data))[0][100])
        plt.axis('off')
        st.pyplot()

    # Close the box around the images
    st.markdown('</div>', unsafe_allow_html=True)
