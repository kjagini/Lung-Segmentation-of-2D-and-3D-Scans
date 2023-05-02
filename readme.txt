README
To run the application follow the commands below:
python -m venv myenv
source myenv/bin/activate      # On Windows, use "myenv\Scripts\activate"
pip install -r requirements.txt
streamlit run Lung_Seg_App.py


This code uses Convolutional LSTM (ConvLSTM) to perform segmentation on medical images. Specifically, it takes in a 5D input array (with dimensions corresponding to batch size, time steps, height, width, and channels) and predicts the corresponding segmentation output.

Dependencies
This code uses several Python libraries, including:

numpy for numerical computing
pandas for reading in CSV files
matplotlib for data visualization
nibabel for reading in medical image files
skimage for image processing
keras for building the neural network
Code Overview
The main steps of this code are as follows:

Import required libraries
Define the neural network model using keras API
Load pre-trained model weights from a saved file
Load input data from a saved .npy file
Use the pre-trained model to predict segmentation output
Visualize input data and predicted output using matplotlib
Model Architecture
The ConvLSTM model architecture used in this code consists of the following layers:

BatchNormalization layer to normalize input data
3D convolutional layers to extract features from input data
Bidirectional ConvFF2D layers to incorporate information from multiple time steps
3D convolutional layers to further extract features from input data
Cropping3D layer to avoid skewing boundaries
ZeroPadding3D layer to pad input data
Sigmoid activation function to produce segmentation output
Input Data
The input data used in this code is a 5D numpy array with dimensions corresponding to batch size, time steps, height, width, and channels.

Output
The output of this code is a segmented image produced by the pre-trained ConvLSTM model. The code visualizes both the input data and predicted output using matplotlib.