import numpy as np
from scipy.signal import resample
from scipy.signal import welch, stft, find_peaks
from scipy.stats import entropy, kurtosis, skew
import pywt
from pywt import wavedec
from antropy import sample_entropy


def resample_eeg_data(eeg_data, new_length):
    num_subjects, num_channels, original_length = eeg_data.shape
    resampled_data = np.zeros((num_subjects, num_channels, new_length))

    for subject_idx in range(num_subjects):
        for channel_idx in range(num_channels):
            signal = eeg_data[subject_idx, channel_idx, :]
            resampled_signal = resample(signal, new_length)
            resampled_data[subject_idx, channel_idx, :] = resampled_signal
    
    return resampled_data

def compute_energy(signal):
    return np.sum(signal ** 2)

def compute_variance(signal):
    return np.var(signal)

def compute_mean(signal):
    return np.mean(signal)

def compute_rms(signal):
    return np.sqrt(np.mean(signal ** 2))

def compute_zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum()

def compute_kurtosis(signal):
    return kurtosis(signal)

def compute_skewness(signal):
    return skew(signal)

def compute_spectral_energy(psd, freqs, band):
    band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.sum(psd[band_idx])

def compute_spectral_entropy(psd):
    psd_normalized = psd / np.sum(psd)  # normalization PSD
    return entropy(psd_normalized)  

def compute_sample_entropy(signal):
    return sample_entropy(signal)
    
def calculate_stft(signal, fs):
    nperseg = 512  
    noverlap = 256  
    # nfft = max(5120, 2 ** np.ceil(np.log2(nperseg)))  
    f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx) 

def calculate_wavelet(signal, wavelet='db4', level=5):
    coeffs = wavedec(signal, wavelet, level=level)
    coeffs_flatten = np.hstack(coeffs)
    return coeffs_flatten

def compute_wavelet_features(data, wavelet='db4'):
    # 定义小波变换的频段
    freq_bands = [(0.5, 4), (4, 8), (8, 16), (16, 32)]
    # 小波变换得到的系数数量
    levels = len(freq_bands)
    
    # 使用小波分解
    coeffs = pywt.wavedec(data, wavelet, level=levels)
    
    features = []
    for i, (low_freq, high_freq) in enumerate(freq_bands):
        coeff = coeffs[i+1]  # 获取对应频段的小波系数
        mean = np.mean(coeff)
        energy_mean = np.mean(np.square(coeff))
        std_dev = np.std(coeff)
        features.extend([mean, energy_mean, std_dev])
    print(len(features))
    return features

def extract_features(eeg_data, wavelet='db4'):
    n_samples, n_channels, n_timepoints = eeg_data.shape
    all_features = np.zeros((n_samples, n_channels, 12))
    
    for sample in range(n_samples):
        for channel in range(n_channels):
            channel_data = eeg_data[sample, channel, :]
            features = compute_wavelet_features(channel_data, wavelet)
            # 123
            all_features[sample, channel, :] = features
    
    return all_features
