import numpy as np
import pandas as pd
import librosa
import os

from AudioAugmentations import *

# Convert csv to dataframe
df = pd.read_csv('speakers_all.csv')

# Display the first 5 rows of the dataframe
df.head()

# Create two classes of data, USA and non-USA, and grab the filenames of both classes
# Also make sure the file_missing? column is False
english = df[(df['native_language'] == 'english') & (df['file_missing?'] == False)]['filename']
non_english = df[(df['native_language'] != 'english') & (df['file_missing?'] == False)]['filename']

# Create a list of countries that have less than 10 speakers
native_languages = df['native_language'].value_counts()
native_languages = native_languages[native_languages < 20].index.tolist()

# Remove the countries that only have 1 speaker
df = df[~df['native_language'].isin(native_languages)]

english = df[(df['native_language'] == 'english') & (df['file_missing?'] == False)]['filename']
non_english = df[(df['native_language'] != 'english') & (df['file_missing?'] == False)]['filename']

# Randomly remove files from non_english until the number of files is equal to the number of english files
non_english = non_english.sample(n=len(english), random_state=42)


# Take n samples from each class and use them to create a validation set
english_validation = english.sample(n=125, random_state=42)
non_english_validation = non_english.sample(n=125, random_state=42)

# Remove the validation samples from the training set
english = english.drop(english_validation.index)
non_english = non_english.drop(non_english_validation.index)

val_set = pd.concat([english_validation, non_english_validation])
train_set = pd.concat([english, non_english])

# Calculate the spectrogram for all audio files in the english class
train_set_spectro = []
train_set_labels = []
i = 0
for file in train_set:
    # Check if file_name exists, if it doesn't, skip it
    file_name = 'recordings/recordings/' + file + '.mp3'
    if not os.path.exists(file_name):
        continue

    spectrogram = calculate_log_mel_spectrogram('recordings/recordings/' + file + '.mp3')
    train_set_spectro.append(spectrogram)

    # Create a pitch shifted version of the spectrogram, first find the sampling rate
    signal, sample_rate = librosa.load('recordings/recordings/' + file + '.mp3')
    pitch_shifted_spectrogram = calculate_log_mel_spectrogram_no_load(signal=pitch_shift(signal, sample_rate, 2), sample_rate=sample_rate)
    train_set_spectro.append(pitch_shifted_spectrogram)

    # Create another pitch shifted version of the spectrogram
    pitch_shifted_spectrogram = calculate_log_mel_spectrogram_no_load(signal=pitch_shift(signal, sample_rate, -2), sample_rate=sample_rate)
    train_set_spectro.append(pitch_shifted_spectrogram)

    # Create a noisy version of the spectrogram
    noisy_spectrogram = calculate_log_mel_spectrogram_no_load(signal=add_background_noise(signal, 0.005), sample_rate=sample_rate)
    train_set_spectro.append(noisy_spectrogram)

    # Create a noisy pitch shifted version of the spectrogram
    noisy_pitch_shifted_spectrogram = calculate_log_mel_spectrogram_no_load(signal=add_background_noise(pitch_shift(signal, sample_rate, 2), 0.005), sample_rate=sample_rate)
    train_set_spectro.append(noisy_pitch_shifted_spectrogram)

    # Create another noisy pitch shifted version of the spectrogram
    noisy_pitch_shifted_spectrogram = calculate_log_mel_spectrogram_no_load(signal=add_background_noise(pitch_shift(signal, sample_rate, -2), 0.005), sample_rate=sample_rate)
    train_set_spectro.append(noisy_pitch_shifted_spectrogram)

    # If the file name has english in it, it's an english speaker, otherwise it's a non-english speaker
    if 'english' in file:
        train_set_labels.append(1)
        train_set_labels.append(1)
        train_set_labels.append(1)
        train_set_labels.append(1)
        train_set_labels.append(1)
        train_set_labels.append(1)
    else:
        train_set_labels.append(0)
        train_set_labels.append(0)
        train_set_labels.append(0)
        train_set_labels.append(0)
        train_set_labels.append(0)
        train_set_labels.append(0)

    # Add a counter to keep track of progress
    i += 1
    if i % 100 == 0:
        print(str(i) + " out of " + str(len(train_set)) + " files processed.")

# Grab the spectros for all audio files in the non-english class
val_set_spectro = []
val_set_labels = []
i = 0
for file in val_set:
    # Check if file_name exists, if it doesn't, skip it
    file_name = 'recordings/recordings/' + file + '.mp3'
    if not os.path.exists(file_name):
        continue

    spectrogram = calculate_log_mel_spectrogram('recordings/recordings/' + file + '.mp3')

    val_set_spectro.append(spectrogram)

    # No need to apply data augmentation to the validation set, add correct label
    if 'english' in file:
        val_set_labels.append(1)
    else:
        val_set_labels.append(0)

    # Add a counter to keep track of progress
    i += 1
    if i % 100 == 0:
        print(str(i) + " out of " + str(len(val_set)) + " files processed.")


# Calculate the minimum length of all spectrograms
min_length = min([spectrogram.shape[1] for spectrogram in train_set_spectro + val_set_spectro])

# Truncate each spectrogram in the english_spectro and non_english_spectro lists to the minimum length
train_set_spectro = [truncate_spectrogram(spectrogram, min_length) for spectrogram in train_set_spectro]
val_set_spectro = [truncate_spectrogram(spectrogram, min_length) for spectrogram in val_set_spectro]

# Normalize each spectrogram in the english_spectro and non_english_spectro lists using min-max normalization
train_set_spectro = [min_max_normalize_spectrogram(spectrogram) for spectrogram in train_set_spectro]
val_set_spectro = [min_max_normalize_spectrogram(spectrogram) for spectrogram in val_set_spectro]

# Add channel dimension as the first dimension to each spectrogram
train_set_spectro = [np.expand_dims(spectrogram, axis=0) for spectrogram in train_set_spectro]
val_set_spectro = [np.expand_dims(spectrogram, axis=0) for spectrogram in val_set_spectro]

# Convert the lists to numpy arrays
train_set_spectro = np.array(train_set_spectro)
val_set_spectro = np.array(val_set_spectro)

# Convert the labels to numpy arrays
train_set_labels = np.array(train_set_labels)
val_set_labels = np.array(val_set_labels)

# Print shapes of first and last spectrogram in both lists
print(train_set_spectro[0].shape)
print(train_set_spectro[-1].shape)

print(val_set_spectro[0].shape)
print(val_set_spectro[-1].shape)

# Print the shape of the arrays
print("Size of training set: ", train_set_spectro.shape)
print("Size of validation set: ", val_set_spectro.shape)

# Count the number of english and non-english speakers in the training and validation sets
print("Number of english speakers in training set: ", np.sum(train_set_labels))
print("Number of non-english speakers in training set: ", len(train_set_labels) - np.sum(train_set_labels))

print("Number of english speakers in validation set: ", np.sum(val_set_labels))
print("Number of non-english speakers in validation set: ", len(val_set_labels) - np.sum(val_set_labels))

file_name = 'log_mel_spectro_data_min_max_norm_augmented.npz'
print("Saving augmented spectrogram data to file: ", file_name)
np.savez(file_name, X_train=train_set_spectro, y_train=train_set_labels, X_val=val_set_spectro, y_val=val_set_labels)


