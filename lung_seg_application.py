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
    st.write("Original Image")
    plt.figure(figsize=(8, 8))

    plt.imshow(loaded_data[0][100])
    st.pyplot()

    # Run the prediction
    prediction = get_the_prediction((loaded_data))

    # Display the prediction
    st.write("Predicted Image")
    plt.figure(figsize=(8, 8))

    plt.imshow(prediction[0][100])
    st.pyplot()
