import socket
import time
import struct
import json
import os

import numpy as np

from utils import *
from feature_extraction import *
from torcheeg.models import DGCNN
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
def standardize(data, mean, std):

    data_expanded = np.expand_dims(data, axis=0)  
    standardized_data = (data_expanded - mean) / std
    standardized_data = np.squeeze(standardized_data, axis=0) 
    
    return standardized_data


# define the related parameters
ip = "127.0.0.1"
port = 12345
option = "predict"
length = 8

# accumulate the recieved data
collected_data = []

# create a socket object
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_address = (ip, port)
sock.bind(server_address)

# Display socket attributes
print('--------------------')
print("-- UDP LISTENER -- ")
print('--------------------')
print("IP:", ip)
print("PORT:", port)
print('--------------------')
print("%s option selected" % option)

# Receive messages
print("Listening...")
start = time.time()
numSamples = 0
duration = -1

# calculate the mean and std from our dataset to standardize the new data
mean, std = get_mean_and_std('E:\Extracted_feature_base')

BASELINE_MEAN = np.array([...])
def predict_movement(data, movement):
    # load the model

    # decode data and append it to the collected data, shape->(4,N)
    data = json.loads(data.decode()).get('data')
    collected_data.append(np.array(data))
    # calculate the templates
    path = 'E:/EEG2feature_base'
    templates = calculate_templates(path)
    mean = 0
    std = 1

    # ensure the loss of data is not important to the prediction
    if len(collected_data) >= 77:
        collected_data = collected_data[-77:]

        # remove the last element of the data
        collected_data[-1] = collected_data[-1][:, :-1]

        # concatenate the data to shape->(4, 384)
        concatenated_data = np.concatenate(collected_data, axis=1)
        
        # expand the data to shape->(1, 4, 384)
        expanded_data = np.expand_dims(concatenated_data, axis=0)

        # extract the features
        temporal_features = extract_temporal_features(data)
        spectral_features = extract_spectral_features(data)
        entropy_features = extract_entropy_features(data)
        template_features = extract_template_features(data, templates)

        # Concatenate features along the fourth dimension
        combined_features = np.concatenate((temporal_features, spectral_features, entropy_features, template_features), axis=2)

        # standardize the data(need to revise the mean and std)
        combined_features = standardize(combined_features, mean, std)
        
        # ensure the shape is correct
        assert combined_features.shape == (1, 4, 23)

        # Convert data to tensors
        combined_features = torch.tensor(combined_features, dtype=torch.float32).squeeze(1)

        model = DGCNN(in_channels=23, num_electrodes=4, hid_channels=64, num_classes=2)

        # according to the movement, load the model
        if movement == "up":
            checkpoint = torch.load('./models/up.ckpt')
        elif movement == "down":
            checkpoint = torch.load('./models/down.ckpt')
        elif movement == "left":
            checkpoint = torch.load('./models/left.ckpt')
        elif movement == "right":
            checkpoint = torch.load('./models/right.ckpt')


        model.load_state_dict(checkpoint['state_dict'])

        # predict the movement
        model.eval()
        with torch.no_grad():
            output = model(combined_features)
            prediction = torch.argmax(output, dim=1).item()
            print(prediction)

        final_prediction = prediction[0].item()

        result = True if final_prediction == 1 else False
        # the result reprents whether the movement is detected or not
        # take "up" as an example, if the model detected the movement is "up", then the result is True
        return result

############################################################################################################
# !!!!! need to revise the selected movement, because I don't know how to get the movement from Unity user interface
while time.time() <= start + duration or duration == -1:
    data, addr = sock.recvfrom(20000) 
    if option == "print":
        print_message(data)
    elif option == "predict":
        result = predict_movement(data, "up")
    numSamples += 1

print("Samples == {}".format(numSamples))
print("Duration == {}".format(duration))
print("Avg Sampling Rate == {}".format(numSamples / duration))