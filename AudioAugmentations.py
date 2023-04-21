import librosa
import numpy as np

# Helper function to calculate the spectrogram
def calculate_spectrogram(audio_file, n_fft=512, hop_length=256):
    signal, sample_rate = librosa.load(audio_file)
    spectrogram = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    
    return np.abs(spectrogram)

def truncate_spectrogram(spectrogram, fixed_length):
    truncated_spectrogram = spectrogram[:, :fixed_length]
    return truncated_spectrogram

def calculate_mel_spectrogram(audio_file, n_fft=512, hop_length=256, n_mels=128):
    signal, sample_rate = librosa.load(audio_file)
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    return mel_spectrogram

def calculate_mel_spectrogram_no_load(signal, sample_rate, n_fft=512, hop_length=256, n_mels=128):
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    return mel_spectrogram

def calculate_log_mel_spectrogram(audio_file, n_fft=512, hop_length=256, n_mels=128):
    signal, sample_rate = librosa.load(audio_file)
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    
    return log_mel_spectrogram

def calculate_log_mel_spectrogram_no_load(signal, sample_rate, n_fft=512, hop_length=256, n_mels=128):
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    
    return log_mel_spectrogram

def normalize_spectrogram(spectrogram):
    mean = np.mean(spectrogram)
    std_dev = np.std(spectrogram)
    normalized_spectrogram = (spectrogram - mean) / std_dev
    return normalized_spectrogram

def min_max_normalize_spectrogram(spectrogram):
    min_value = np.min(spectrogram)
    max_value = np.max(spectrogram)
    normalized_spectrogram = (spectrogram - min_value) / (max_value - min_value)
    return normalized_spectrogram

def pitch_shift(audio_data, sample_rate, pitch_shift_steps):
    return librosa.effects.pitch_shift(y=audio_data, sr=sample_rate, n_steps=pitch_shift_steps)

def add_background_noise(audio_data, noise_amplitude):
    noise = np.random.normal(0, noise_amplitude, len(audio_data))
    return audio_data + noise

def extract_mfcc(audio_file, n_mfcc=13, n_fft=512, hop_length=256):
    signal, sample_rate = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc

def pad_spectrogram(spectrogram, target_length, noise=True, noise_scale=0.003):
    if noise:
        padding = np.random.normal(loc=0, scale=noise_scale, size=(spectrogram.shape[0], target_length - spectrogram.shape[1]))
    else:
        padding = np.full((spectrogram.shape[0], target_length - spectrogram.shape[1]), 0.0)

    padded_spectrogram = np.hstack((spectrogram, padding))
    return padded_spectrogram

