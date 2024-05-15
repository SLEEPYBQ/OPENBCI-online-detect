import h5py
import os
import glob
import numpy as np
from scipy.signal import resample

import socket
import time
import struct
import json


def load_h5py_data(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['data'][()]
        labels = f['labels'][()]
    return data, labels

def save_features_to_h5(data, labels, filepath):
    with h5py.File(filepath, 'w') as h5file:
        h5file.create_dataset('data', data=data)
        h5file.create_dataset('labels', data=labels)

def get_data(path):
    for filepath in glob.glob(os.path.join(path, 'A*.h5')):
        # Load data from file
        data, labels = load_h5py_data(filepath)
        print(f"Loaded data from {filepath}")
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")

def get_source_data(path):
    # Initialize lists to store all data and labels
    all_data = []
    all_labels = []
    
    # Iterate over all .h5 files matching the pattern A%d.h5 in the specified directory
    for filepath in glob.glob(os.path.join(path, 'A*.h5')):
        # Load data from file
        data, labels = load_h5py_data(filepath)

        print('filepath:', filepath)
        
        # Remove singleton dimensions and expand dims as needed
        data = np.squeeze(data, axis=1)
        data = np.expand_dims(data, axis=1)
        
        # Append data and labels to the lists
        all_data.append(data)
        all_labels.append(labels)
    print('all_data:', all_data)
    
    # Concatenate all data and labels into numpy arrays
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Shuffle the dataset
    shuffle_indices = np.random.permutation(len(all_data))
    all_data = all_data[shuffle_indices]
    all_labels = all_labels[shuffle_indices]
    
    # Split dataset into training and testing sets
    split_index = int(0.7 * len(all_data))
    train_data = all_data[:split_index]
    train_labels = all_labels[:split_index]
    test_data = all_data[split_index:]
    test_labels = all_labels[split_index:]
    
    # Standardize the training and testing data
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    return train_data, train_labels, test_data, test_labels


def load_and_save_subject_data(root_dir):
    subject_number = 1  # Start numbering from 1

    # Loop through each subject folder
    for subject_folder in sorted(os.listdir(root_dir)):
        print(f"Processing subject: {subject_folder}")
        if os.path.isdir(os.path.join(root_dir, subject_folder)):
            data_dir = os.path.join(root_dir, subject_folder, 'sliced_data')
            labels = {'up_slices.npy': 0, 'down_slices.npy': 1, 'left_slices.npy': 2, 'right_slices.npy': 3, 'baseline_slices.npy':4 }

            all_data = []
            all_labels = []

            # Load each task file and assign labels
            for file_name, label in labels.items():
                file_path = os.path.join(data_dir, file_name)
                if os.path.exists(file_path):
                    data = np.load(file_path)
                    # Add a new axis to represent convolutional channels, and transpose the dimensions
                    data = np.expand_dims(data, axis=1)  # (trials, 1, electrodes, samples)
                    all_data.append(data)
                    all_labels.extend([label] * data.shape[0])

            # Concatenate all data and labels
            if all_data:
                all_data = np.concatenate(all_data, axis=0)
                all_labels = np.array(all_labels)

                # Shuffle the data
                indices = np.random.permutation(len(all_data))
                all_data = all_data[indices]
                all_labels = all_labels[indices]

                # Standardize the data
                mean = np.mean(all_data, axis=(0, 2, 3), keepdims=True)
                std = np.std(all_data, axis=(0, 2, 3), keepdims=True)
                all_data = (all_data - mean) / std

                # Save the data for each subject using the A%d.h5 format
                filename = os.path.join(root_dir, f'A{subject_number}.h5')
                with h5py.File(filename, 'w') as hdf:
                    hdf.create_dataset('data', data=all_data)
                    hdf.create_dataset('labels', data=all_labels)

                subject_number += 1  # Increment the subject number for the next valid subject


def slice_eeg_action(data, events, event_ids, max_length=3000, save_path="sliced_data", sampling_rate=1000,
                     target_rate=128):
    
    baseline_mean = None
    if not os.path.exists(save_path):
        os.makedirs(save_path)  


    new_length = int(max_length * target_rate / sampling_rate)

    baseline_start_idx = np.where(events[:, 2] == 11)[0]
    baseline_end_idx = np.where(events[:, 2] == 12)[0]
    if baseline_start_idx.size > 0 and baseline_end_idx.size > 0:
        baseline_data = data[:, events[baseline_start_idx[0], 0]:events[baseline_end_idx[0], 0]]
        baseline_mean = np.mean(baseline_data, axis=1)
        print(f"Baseline mean calculated. Shape: {baseline_mean.shape}")
        print(f"Baseline mean: {baseline_mean}")
    slices = {}
    counts = {}
    for label, (start_id, end_id) in event_ids.items():
        start_idx = np.where(events[:, 2] == start_id)[0]
        end_idx = np.where(events[:, 2] == end_id)[0]
        correct_idx = np.where((events[:, 2] == 15) | (events[:, 2] == 16))[0]  

        slices[label] = []
        for start, end in zip(start_idx, end_idx):
            if start < end:
                keys = correct_idx[(correct_idx > end) & (correct_idx < end + 1)]  
                if any(events[keys, 2] == 16):
                    continue  

                start_sample = events[start, 0]
                if events[end, 0] - start_sample < max_length:
                    end_sample = start_sample + max_length
                else:
                    end_sample = min(events[end, 0], start_sample + max_length) 

                data_slice = data[:, start_sample:end_sample]
                data_slice = data_slice - baseline_mean[:, None]
                data_slice = resample(data_slice, new_length, axis=1)


                slices[label].append(data_slice)

        counts[label] = len(slices[label])
        file_path = os.path.join(save_path, f"{label}_slices.npy")
        np.save(file_path, slices[label])
        print(f"Saved {len(slices[label])} slices of '{label}' to '{file_path}'")

    return counts


def slice_baseline(data, events, save_path="baseline_slices",  max_length=3000, sampling_rate=1000, slice_duration=3, min_valid_duration=1, target_rate=128):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    
    slice_length = int(slice_duration * sampling_rate)
    min_valid_length = int(min_valid_duration * sampling_rate)
    new_length = int(max_length * target_rate / sampling_rate)
    
    
    baseline_start_idx = np.where(events[:, 2] == 11)[0]
    baseline_end_idx = np.where(events[:, 2] == 12)[0]
    
    if baseline_start_idx.size == 0 or baseline_end_idx.size == 0:
        raise ValueError("Baseline start or end event not found in the events array.")
    
    
    baseline_data = data[:, events[baseline_start_idx[0], 0]:events[baseline_end_idx[0], 0]]
    baseline_length = baseline_data.shape[1]
    
    slices = []
    
    for start in range(0, baseline_length, slice_length):
        end = start + slice_length
        if end > baseline_length:
            slice_data = np.zeros((baseline_data.shape[0], slice_length))
            valid_length = baseline_length - start
            if valid_length < min_valid_length:
                continue
            slice_data[:, :baseline_length - start] = baseline_data[:, start:baseline_length]
        else:
            slice_data = baseline_data[:, start:end]
        
        slice_data = resample(slice_data, new_length, axis=1)
        slices.append(slice_data)
    
    file_path = os.path.join(save_path, "baseline_slices.npy")
    np.save(file_path, slices)
    print(f"Saved {len(slices)} baseline slices to '{file_path}'")
    
    return len(slices)


def calculate_templates(path):
    all_data = []
    all_labels = []
    
    for filepath in glob.glob(os.path.join(path, 'A*.h5')):
        data, labels = load_h5py_data(filepath)
        # print(f"Data Shape: {data.shape}, Labels Shape: {labels.shape}")
        # print(f"Data Stats: Min={np.min(data)}, Max={np.max(data)}, Mean={np.mean(data)}")
        if data.ndim > 2:
            data = np.squeeze(data)  # Ensure it matches expected dimensions
        all_data.append(data)
        all_labels.append(labels)
    
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    templates = {}
    movements = [0, 1, 2, 3]  # Labels corresponding to up, down, left, right
    for movement in movements:
        movement_data = all_data[all_labels == movement]
        if movement_data.size == 0:
            print(f"No data for movement {movement}.")
            continue
        templates[movement] = np.mean(movement_data, axis=0)

    if all_data.size > 0:
        templates['combined'] = np.mean(all_data, axis=0)
    else:
        print("No data available to calculate combined template.")
    
    return templates


def get_source_data_sub(path):
    # Initialize lists to store all data and labels
    train_data = []
    train_labels = []

    test_data = []
    test_labels = []
    
    # Iterate over all .h5 files matching the pattern A%d.h5 in the specified directory
    for filepath in glob.glob(os.path.join(path, 'F*E.h5')):
        # Load data from file
        data, labels = load_h5py_data(filepath)

        print('filepath:', filepath)
        
        # Remove singleton dimensions and expand dims as needed
        data = np.squeeze(data, axis=1)
        data = np.expand_dims(data, axis=1)
        
        # Append data and labels to the lists
        train_data.append(data)
        train_labels.append(labels)
    # print('all_data:', all_data)

    for filepath in glob.glob(os.path.join(path, 'F*T.h5')):
        # Load data from file
        data, labels = load_h5py_data(filepath)

        print('filepath:', filepath)
        
        # Remove singleton dimensions and expand dims as needed
        data = np.squeeze(data, axis=1)
        data = np.expand_dims(data, axis=1)
        
        # Append data and labels to the lists
        test_data.append(data)
        test_labels.append(labels)
    
    # Concatenate all data and labels into numpy arrays
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    # Shuffle the dataset
    shuffle_indices = np.random.permutation(len(train_data))
    train_data = train_data[shuffle_indices]
    train_labels = train_labels[shuffle_indices]

    shuffle_indices = np.random.permutation(len(test_data))
    test_data = test_data[shuffle_indices]
    test_labels = test_labels[shuffle_indices]
    

    # Standardize the training and testing data
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    return train_data, train_labels, test_data, test_labels


from torcheeg.transforms.pyg import ToG
from torcheeg.datasets.constants import T_ADJACENCY_MATRIX
def convert_to_graph(dataset):
    t = ToG(adj=T_ADJACENCY_MATRIX)
    graph_data = []
    for data, label in dataset:
        graph_data.append((t(eeg=data)['eeg'], label))
    return graph_data



def get_source_data_train(path):
    # Initialize lists to store all data and labels
    all_data = []
    all_labels = []
    
    # Iterate over all .h5 files matching the pattern A%d.h5 in the specified directory
    for filepath in glob.glob(os.path.join(path, 'F*.h5')):
        # Load data from file
        data, labels = load_h5py_data(filepath)

        print('filepath:', filepath)
        
        # Remove singleton dimensions and expand dims as needed
        data = np.squeeze(data, axis=1)
        data = np.expand_dims(data, axis=1)
        
        # Append data and labels to the lists
        all_data.append(data)
        all_labels.append(labels)
    # print('all_data:', all_data)
    
    # Concatenate all data and labels into numpy arrays
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Shuffle the dataset
    shuffle_indices = np.random.permutation(len(all_data))
    all_data = all_data[shuffle_indices]
    all_labels = all_labels[shuffle_indices]
    
    # Split dataset into training and testing sets
    split_index = int(0.7 * len(all_data))
    train_data = all_data[:split_index]
    train_labels = all_labels[:split_index]
    test_data = all_data[split_index:]
    test_labels = all_labels[split_index:]
    
    # Standardize the training and testing data
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    return train_data, train_labels, test_data, test_labels


def get_source_data_detect(path):
    # Initialize lists to store all data and labels
    all_data = []
    all_labels = []
    
    # Iterate over all .h5 files matching the pattern F*.h5 in the specified directory
    for filepath in glob.glob(os.path.join(path, 'F*.h5')):
        # Load data from file
        data, labels = load_h5py_data(filepath)

        print('filepath:', filepath)
        
        # Remove singleton dimensions and expand dims as needed
        data = np.squeeze(data, axis=1)
        data = np.expand_dims(data, axis=1)
        
        # Append data and labels to the lists
        all_data.append(data)
        all_labels.append(labels)
    
    # Concatenate all data and labels into numpy arrays
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Shuffle the dataset
    shuffle_indices = np.random.permutation(len(all_data))
    all_data = all_data[shuffle_indices]
    all_labels = all_labels[shuffle_indices]
    
    # Standardize the training and testing data
    # mean = np.mean(all_data)
    # std = np.std(all_data)
    # all_data = (all_data - mean) / std

    mean = np.mean(all_data, axis=0, keepdims=True)
    std = np.std(all_data, axis=0, keepdims=True)
    all_data = (all_data - mean) / std

    
    return all_data, all_labels

def get_mean_and_std(path):
    # Initialize lists to store all data and labels
    all_data = []
    all_labels = []
    
    # Iterate over all .h5 files matching the pattern F*.h5 in the specified directory
    for filepath in glob.glob(os.path.join(path, 'F*.h5')):
        # Load data from file
        data, labels = load_h5py_data(filepath)

        print('filepath:', filepath)
        
        # Remove singleton dimensions and expand dims as needed
        data = np.squeeze(data, axis=1)
        data = np.expand_dims(data, axis=1)
        
        # Append data and labels to the lists
        all_data.append(data)
        all_labels.append(labels)
    
    # Concatenate all data and labels into numpy arrays
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Shuffle the dataset
    shuffle_indices = np.random.permutation(len(all_data))
    all_data = all_data[shuffle_indices]
    all_labels = all_labels[shuffle_indices]
    
    # Standardize the training and testing data
    mean = np.mean(all_data, axis=0, keepdims=True)
    std = np.std(all_data, axis=0, keepdims=True)
    all_data = (all_data - mean) / std

    print(all_data.shape)

    
    return mean, std
def get_raw_baseline(path):
    # Initialize lists to store all data and labels
    all_data = []
    all_labels = []
    
    # Iterate over all .h5 files matching the pattern A%d.h5 in the specified directory
    for filepath in glob.glob(os.path.join(path, 'A*.h5')):
        # Load data from file
        data, labels = load_h5py_data(filepath)

        print('filepath:', filepath)
        
        # Remove singleton dimensions and expand dims as needed
        data = np.squeeze(data, axis=1)
        data = np.expand_dims(data, axis=1)
        
        # Append data and labels to the lists
        all_data.append(data)
        all_labels.append(labels)
    print('all_data:', all_data)
    
    # Concatenate all data and labels into numpy arrays
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Shuffle the dataset
    shuffle_indices = np.random.permutation(len(all_data))
    all_data = all_data[shuffle_indices]
    all_labels = all_labels[shuffle_indices]
    
    # Split dataset into training and testing sets
    split_index = int(0.7 * len(all_data))
    train_data = all_data[:split_index]
    train_labels = all_labels[:split_index]
    test_data = all_data[split_index:]
    test_labels = all_labels[split_index:]
    
    # Standardize the training and testing data
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    return mean, std


def print_message(data):
    try:
        # print(data) 
        obj = json.loads(data.decode())
        print(obj.get('data'))
    except BaseException as e:
        print(e)