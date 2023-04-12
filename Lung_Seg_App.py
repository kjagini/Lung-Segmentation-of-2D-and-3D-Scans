#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob
from sklearn.model_selection import train_test_split

from skimage.util import montage as montage2d
from conv import ConvRNN2D, ConvFF2DCell, ConvFF2D
from keras.models import model_from_json


# In[ ]:


from keras.layers import ConvLSTM2D, Bidirectional, BatchNormalization, Conv3D, Cropping3D, ZeroPadding3D
from keras.models import Sequential
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
# sim_model.summary()


# In[ ]:





# In[ ]:


import streamlit as st
import cv2
from PIL import Image
import numpy as np

def app():
    st.title("Image Segmentation")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "npy"])
    
    if uploaded_image is not None:
        # Load image
        loaded_data = np.load('prediction_array.npy')
        st.image(loaded_data[0][100], caption="Original Image", use_column_width=True)
        # sim_model.load_weights('convlstm_model_weights.best.hdf5')
        # pred_seg = sim_model.predict(loaded_data)
        st.image(loaded_data[0][100], caption="Original Image", use_column_width=True)
        
        # st.image(pred_seg[0][100], caption="Segmented Image", use_column_width=True)

if __name__ == '__main__':
    app()


# In[ ]:




