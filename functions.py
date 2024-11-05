import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import FeatureExtraction as FE
import tensorflow as tf
from scipy.io import loadmat
from pyvistaqt import BackgroundPlotter
from scipy.signal import welch, stft, find_peaks, resample
from scipy.stats import entropy, kurtosis, skew
from pywt import wavedec
from antropy import sample_entropy
# %matplotlib qt


def load_clean_data(participate : int, event_id : int):
    raw_clean_path = f'clean_data/P{participate:02d}_clean.fif'
    raw_clean = mne.io.read_raw_fif(raw_clean_path, preload=True)
    
    events = mne.find_events(raw_clean, stim_channel='STI 014', shortest_event=0)
#     print (events)
    
    # select the events which less than 245
    event_id_to_select = 245
    selected_events = events[events[:, 2] < event_id_to_select]

#     print(selected_events.shape)
#     print(selected_events)
    
    # extract event data
    event_indices = np.where(selected_events[:, 2] == event_id)[0]
    
    event_data_list = []
    
    for i in event_indices:
        start_sample = selected_events[i, 0]
        if i < len(selected_events) - 1:
            end_sample = selected_events[i + 1, 0]
        else:
            end_sample = len(raw_clean.times)
        
        event_segment = raw_clean[:, start_sample:end_sample][0]
        event_data_list.append(event_segment)
        
    # check the data of event ID
#     print(f'Event ID: {event_id}, Number of segments: {len(event_data_list)}')
#     for i, segment in enumerate(event_data_list):
#         print(f'Segment {i + 1}: shape = {segment.shape}')

    # calculate the median length
    segment_lengths = [seg.shape[1] for seg in event_data_list]
    median_length = int(np.median(segment_lengths))
#     print(f'Median length: {median_length}')

    # resample the segmentation
    resampled_segments = []
    for seg in event_data_list:
        resampled_seg = resample(seg, median_length, axis=1)
        resampled_segments.append(resampled_seg)

    # transfer to an array
    resampled_array = np.array(resampled_segments)
#     print(f'Resampled array shape: {resampled_array.shape}')
    resampled_array1 = resampled_array[:, :64, :]
    return resampled_array1
    
def load_data(ids: int):
    array1 = load_clean_data(1,ids)
    array4 = load_clean_data(4,ids)
    array5 = load_clean_data(5,ids)
    array6 = load_clean_data(6,ids)
    array7 = load_clean_data(7,ids)
    array9 = load_clean_data(9,ids)
    array11 = load_clean_data(11,ids)
    array12 = load_clean_data(12,ids)
    array13 = load_clean_data(13,ids)
    array14 = load_clean_data(14,ids)

    new_length = 10240
    array1 = FE.resample_eeg_data(array1, new_length)
    array4 = FE.resample_eeg_data(array4, new_length)
    array5 = FE.resample_eeg_data(array5, new_length)
    array6 = FE.resample_eeg_data(array6, new_length)
    array7 = FE.resample_eeg_data(array7, new_length)
    array9 = FE.resample_eeg_data(array9, new_length)
    array11 = FE.resample_eeg_data(array11, new_length)
    array12 = FE.resample_eeg_data(array12, new_length)
    array13 = FE.resample_eeg_data(array13, new_length)
    array14 = FE.resample_eeg_data(array14, new_length)

    resample = np.vstack((array1, array4, array5, array6, array7, array9, array11, array12, array13, array14))
    print(resample.shape)
    filename = f'resample_EEG/resample_EEG_song{ids}.npy'
    np.save(filename, resample)

    return f'resample_EEG_song{ids}.npy is already saved successfully.'
    
import numpy as np
from tensorflow.keras.utils import Sequence

class PerClassAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.per_class_accuracy = {0: [], 1: [], 2: []}

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val)

        # 转换预测结果为类别索引
        y_true = np.argmax(y_val, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # 计算每个类别的准确率
        for i in range(3):
            idx = np.where(y_true == i)[0]
            acc = np.sum(y_pred_classes[idx] == y_true[idx]) / len(idx)
            self.per_class_accuracy[i].append(acc)
            print(f'Accuracy for class {i+1} (epoch {epoch+1}): {acc:.4f}')

    def get_per_class_accuracy(self):
        return self.per_class_accuracy

class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = self.X[indexes]
        y_batch = self.y[indexes]
        return X_batch, y_batch

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def feature_extraction(array):
    num_samples = 5
    fs = 512 
    num_channels = array.shape[1]
    num_features_per_channel = 12  
    # stft_feature_length = 128 * 39  
    # wavelet_feature_length = 256  
    features = np.zeros((num_samples, num_channels, num_features_per_channel))

    low_band = (0.5, 4)
    mid_band = (4, 8)
    high_band = (8, 13)

    for sample_idx in range(num_samples):
        for channel_idx in range(num_channels):
            signal = array[sample_idx - 1, channel_idx - 1, :]
            raw.filter(l_freq=0.5, h_freq=4)
            raw.filter(l_freq=4, h_freq=8)
            raw.filter(l_freq=8, h_freq=13)
            raw.filter(l_freq=13, h_freq=35)

            # time domain
            energy = FE.compute_energy(signal)
            variance = FE.compute_variance(signal)
            mean_value = FE.compute_mean(signal)
            rms_value = FE.compute_rms(signal)
            zero_crossing_rate = FE.compute_zero_crossing_rate(signal)
            kurtosis_value = FE.compute_kurtosis(signal)
            skewness_value = FE.compute_skewness(signal)
            sample_ent = FE.compute_sample_entropy(signal)

            # PSD
            freqs, psd = welch(signal, fs=fs, nperseg=512, noverlap = 256)
            
            low_energy = FE.compute_spectral_energy(psd, freqs, low_band)
            mid_energy = FE.compute_spectral_energy(psd, freqs, mid_band)
            high_energy = FE.compute_spectral_energy(psd, freqs, high_band)

            spec_entropy = FE.compute_spectral_entropy(psd)

            # stft
    #         stft_features = compute_stft_features(signal, fs)

            # wavelet
    #         wavelet_features = FE.compute_wavelet(signal)


            # merge
            features[sample_idx, channel_idx, :] = [
                energy, variance, mean_value, rms_value, zero_crossing_rate, 
                kurtosis_value, skewness_value, low_energy, mid_energy, 
                high_energy, spec_entropy, sample_ent
            ]
    #         sample_features.extend(stft_features)
    #         sample_features.extend(wavelet_features)


        # features[sample_idx, :] = np.array(sample_features)
    
    return features

def extract_stft_wavelet(eeg_data, fs=512, wavelet='db4', level=5):
    num_subjects, num_channels, num_samples = eeg_data.shape
    # Example calculation to determine the feature length
    example_stft = FE.calculate_stft(eeg_data[0, 0, :], fs)
    example_wavelet = FE.calculate_wavelet(eeg_data[0, 0, :], wavelet, level)
    
    stft_length = example_stft.shape[0] * example_stft.shape[1]
    wavelet_length = len(example_wavelet)
    total_features_length = stft_length + wavelet_length
    
    features = np.zeros((num_subjects, num_channels, total_features_length))
    
    for subject_idx in range(num_subjects):
        for channel_idx in range(num_channels):
            signal = eeg_data[subject_idx, channel_idx, :]
            
            # Calculate STFT features
            stft_features = FE.calculate_stft(signal, fs)
            stft_features_flatten = stft_features.flatten()
            
            # Calculate wavelet features
            wavelet_features = FE.calculate_wavelet(signal, wavelet, level)
            
            # Combine features
            combined_features = np.hstack((stft_features_flatten, wavelet_features))
            
            # Store features
            features[subject_idx, channel_idx, :] = combined_features
            
    return features

def find_most_diff(array1, array2):
    difference = array1 - array2

    # compute RMS
    rms_difference = np.sqrt(np.mean(np.square(difference), axis=1))
    
    # find the indes of most different channels
    top_20_channels = np.argsort(rms_difference)[-20:][::-1]
    
    print(top_20_channels)
    return top_20_channels
    