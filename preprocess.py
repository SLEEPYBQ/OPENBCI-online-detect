from neuracle_lib.readbdfdata import readbdfdata
import numpy as np
import os
from utils import load_h5py_data
from utils import load_and_save_subject_data
from utils import slice_eeg_action
import glob
from scipy.signal import butter, filtfilt
import mne
from utils import slice_baseline


def select_channels(data, channel_names, selected_channels):
    selected_indices = [channel_names.index(name) for name in selected_channels]

    selected_data = data[selected_indices, :]

    return selected_data


def apply_notch_filter(data, notch_freq, fs, quality_factor=30):
    """
    Apply a notch filter to EEG data to remove powerline noise at a specified frequency using a 2nd order Butterworth zero-phase filter.

    Parameters:
    data : ndarray
        The EEG data, shape (channels, samples).
    notch_freq : float
        The frequency at which the notch filter is applied, typically the powerline frequency.
    fs : int
        The sampling frequency of the data.
    quality_factor : float
        The quality factor of the notch filter, which affects the bandwidth around the notch frequency.

    Returns:
    filtered_data : ndarray
        The notch-filtered EEG data.
    """
    # Create an MNE RawArray object
    info = mne.create_info(ch_names=[f'chan{i}' for i in range(data.shape[0])], sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    # Apply notch filter
    raw.notch_filter(notch_freq, notch_widths=notch_freq/quality_factor)
    
    # Return filtered data
    return raw.get_data()

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to EEG data to remove DC offsets and high frequency noise.

    Parameters:
    data : ndarray
        The EEG data, shape (channels, samples).
    lowcut : float
        The lower boundary of the frequency range.
    highcut : float
        The upper boundary of the frequency range.
    fs : int
        The sampling frequency of the data.
    order : int
        The order of the Butterworth filter.

    Returns:
    filtered_data : ndarray
        The bandpass-filtered EEG data.
    """
    # Create an MNE RawArray object
    info = mne.create_info(ch_names=[f'chan{i}' for i in range(data.shape[0])], sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    # Apply bandpass filter
    raw.filter(l_freq=lowcut, h_freq=highcut, method='iir', iir_params={'order': order, 'ftype': 'butter'})
    
    # Return filtered data
    return raw.get_data()



if __name__ == '__main__':
    fs = 1000  # Sampling frequency in Hz
    notch_freq = 50  # Notch filter frequency in Hz

    base_path = "E:/EEG/"
    output_base_path = "E:/Sliced_reg_test/"

    # define the channels to be used
    channel_names = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7',
                     'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4',
                     'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4',
                     'P5', 'P6', 'P7', 'P8', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2', 'ECG',
                     'HEOR', 'HEOL', 'VEOU', 'VEOL']
    selected_channels = ['T7', 'T8', 'C3', 'C4']

    # define the main task index
    main_event_ids = {
        'baseline': (11, 12),
        'real_task': (17, 18),
        'imagine_task': (19, 20),
        'visual_stimulus_task': (37, 38)
    }

    # define the action task index
    action_event_ids = {
        'up': (1, 5),
        'down': (2, 6),
        'left': (3, 7),
        'right': (4, 8),
        'imagine_up': (21, 25),
        'imagine_down': (22, 26),
        'imagine_left': (23, 27),
        'imagine_right': (24, 28),
        'visual_stimulus_up': (29, 33),
        'visual_stimulus_down': (30, 34),
        'visual_stimulus_left': (31, 35),
        'visual_stimulus_right': (32, 36)
    }

    for folder in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, folder)) and folder[:2].isdigit():
            full_path = os.path.join(base_path, folder)
            filenames = ['data.bdf', 'evt.bdf']
            pathname = [full_path, full_path]

            eeg = readbdfdata(filenames, pathname)
            data = eeg['data']
            events = eeg['events']

            selected_data = select_channels(data, channel_names, selected_channels)
            

            # Apply the filters
            filtered_data = apply_notch_filter(selected_data, notch_freq, fs)
            final_filtered_data = apply_bandpass_filter(filtered_data, 0.1, 45, fs)
            # print(final_filtered_data)
            action_event_save_path = os.path.join(output_base_path, folder, 'sliced_data')

            slice_eeg_action(final_filtered_data, events, action_event_ids, 3000, action_event_save_path)
            slice_baseline(final_filtered_data, events, action_event_save_path)



    # save to h5py
    root_dir = "E:/Sliced_reg/"
    load_and_save_subject_data(root_dir)
    print(f"All Done, saved in {root_dir}")


    

