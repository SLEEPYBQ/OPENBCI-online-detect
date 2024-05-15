from utils import get_data
import numpy as np
from scipy.stats import linregress
from utils import load_h5py_data
import glob
import os
from scipy.signal import welch
from pyentrp import entropy as ent
from utils import calculate_templates
from scipy.signal import correlate
from scipy.stats import pearsonr
import nolds
from utils import save_features_to_h5

def extract_temporal_features(data):
    # Assume data shape is (80, 1, 4, 384)
    num_samples, _, num_channels, num_points = data.shape
    # Prepare an output array of shape (80, 1, 4, 6)
    features = np.zeros((num_samples, 1, num_channels, 6))

    # Iterate over each sample and channel to compute the features
    for sample in range(num_samples):
        for channel in range(num_channels):
            channel_data = data[sample, 0, channel, :]
            
            # Compute each feature
            mean_val = np.mean(channel_data)
            min_peak = np.min(channel_data)
            max_peak = np.max(channel_data)
            peak_to_peak = max_peak - min_peak
            slope, intercept, _, _, _ = linregress(np.arange(num_points), channel_data)
            
            # Store the features
            features[sample, 0, channel, 0] = mean_val
            features[sample, 0, channel, 1] = min_peak
            features[sample, 0, channel, 2] = max_peak
            features[sample, 0, channel, 3] = peak_to_peak
            features[sample, 0, channel, 4] = slope
            features[sample, 0, channel, 5] = intercept

    return features

def extract_spectral_features(data, fs=128):
    # Assume data shape is (80, 1, 4, 384)
    num_samples, _, num_channels, num_points = data.shape
    # Prepare an output array of shape (80, 1, 4, 5)
    features = np.zeros((num_samples, 1, num_channels, 5))

    # Define frequency bands
    freq_bands = [(0, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    
    # Iterate over each sample and channel to compute the spectral features
    for sample in range(num_samples):
        for channel in range(num_channels):
            channel_data = data[sample, 0, channel, :]
            f, Pxx = welch(channel_data, fs=fs, window='hann', nperseg=192, noverlap=96)
            
            # Compute power in each frequency band
            band_powers = []
            for low, high in freq_bands:
                mask = (f >= low) & (f <= high)
                mean_power = np.mean(Pxx[mask])
                band_powers.append(mean_power)
            
            features[sample, 0, channel, :] = band_powers

    return features

def extract_entropy_features(data):
    num_samples, _, num_channels, num_points = data.shape
    features = np.zeros((num_samples, 1, num_channels, 2))  # Adjusted for two entropy measures
    
    m = 3  # Embedding dimension for permutation entropy
    sample_length = 2  # Sample length for sample entropy, adjust as needed
    
    # Iterate over each sample and channel to compute the entropy features
    for sample in range(num_samples):
        for channel in range(num_channels):
            channel_data = data[sample, 0, channel, :].flatten()

            # Permutation entropy
            perm_entropy = ent.permutation_entropy(channel_data, order=m, delay=1, normalize=True)
            
            # Sample entropy
            tolerance = 0.1 * np.std(channel_data)  # Default tolerance setting
            sample_entropy_values = ent.sample_entropy(channel_data, sample_length, tolerance)
            
            if len(sample_entropy_values) > 0:
                sample_entropy = sample_entropy_values[0]  # Using the first entropy value
            else:
                sample_entropy = np.nan  # Handle cases where no valid value is returned

            # Store the entropy measures
            features[sample, 0, channel, 0] = perm_entropy
            features[sample, 0, channel, 1] = sample_entropy
    
    return features

def extract_template_features(data, templates):
    # Assume data shape is (80, 1, 4, 384)
    num_samples, _, num_channels, num_points = data.shape
    # Prepare an output array of shape (80, 1, 4, 10)
    features = np.zeros((num_samples, 1, num_channels, 10))
    
    # Template keys assuming labels 0, 1, 2, 3 correspond to up, down, left, right and 'combined'
    movement_keys = [0, 1, 2, 3, 'combined']

    # Iterate over each sample and channel to compute the features
    for sample in range(num_samples):
        for channel in range(num_channels):
            channel_data = data[sample, 0, channel, :]

            for i, key in enumerate(movement_keys):
                template = templates[key][channel, :]  # Corrected indexing for 2D template

                # Pearson correlation between the channel data and the template
                correlation, _ = pearsonr(channel_data, template)
                features[sample, 0, channel, i] = correlation

                # Maximum cross-correlation between the channel data and the template
                xcorr = correlate(channel_data, template, mode='full')
                max_xcorr = np.max(xcorr)
                features[sample, 0, channel, i + 5] = max_xcorr  # Offset by 5 for cross-correlation features

    return features



if __name__ == '__main__':
    path = 'E:/EEG2feature_base'
    output_path = 'E:/Extracted_feature_base'
    templates = calculate_templates(path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)  # Create the directory if it does not exist

    for filepath in glob.glob(os.path.join(path, 'A*.h5')):
        filename = os.path.basename(filepath)
        output_filename = 'F' + filename[1:]  # Change 'A' to 'F' in the filename
        output_filepath = os.path.join(output_path, output_filename)

        # Load data from file
        data, labels = load_h5py_data(filepath)
        temporal_features = extract_temporal_features(data)
        spectral_features = extract_spectral_features(data)
        entropy_features = extract_entropy_features(data)
        template_features = extract_template_features(data, templates)

        # Concatenate features along the fourth dimension
        combined_features = np.concatenate((temporal_features, spectral_features, entropy_features, template_features), axis=3)

        # Save the concatenated features to a new H5 file
        save_features_to_h5(combined_features, labels, output_filepath)
        
        print(f"Features saved to {output_filepath}")
        print(f"Combined Features shape: {combined_features.shape}")