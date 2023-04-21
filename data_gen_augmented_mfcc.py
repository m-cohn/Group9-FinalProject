
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

# Create a helper function for extracting the MFCCs from the audio files
def extract_mfcc_1d(audio_file, n_mfcc=40, sample_rate=16000, load=True):
    if load:
        signal, sample_rate = librosa.load(audio_file)
        mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=512, hop_length=256)
        return np.mean(mfcc, axis=1)
    

    mfcc = librosa.feature.mfcc(y=audio_file, sr=sample_rate, n_mfcc=n_mfcc, n_fft=512, hop_length=256)
    return np.mean(mfcc, axis=1)

# Take n samples from each class and use them to create a validation set
english_validation = english.sample(n=125, random_state=42)
non_english_validation = non_english.sample(n=125, random_state=42)

# Remove the validation samples from the training set
english = english.drop(english_validation.index)
non_english = non_english.drop(non_english_validation.index)

val_set = pd.concat([english_validation, non_english_validation])
train_set = pd.concat([english, non_english])

# Calculate the mfcc for all audio files in the english class
train_set_mfcc = []
train_set_labels = []
i = 0
for file in train_set:
    # Check if file_name exists, if it doesn't, skip it
    file_name = 'recordings/recordings/' + file + '.mp3'
    if not os.path.exists(file_name):
        continue
    
    # Load the audio file
    signal, sample_rate = librosa.load(file_name)
    mfcc = extract_mfcc_1d(file_name, n_mfcc=40)
    train_set_mfcc.append(mfcc)

    # Create a pitch shifted version of the mfcc, first find the sampling rate
    # pitch_shifted_mfcc = librosa.feature.mfcc(y=pitch_shift(signal, sample_rate, 2), sr=sample_rate, n_mfcc=40)
    pitch_shifted_mfcc = extract_mfcc_1d(pitch_shift(signal, sample_rate, 2), n_mfcc=40, load=False, sample_rate=sample_rate)
    train_set_mfcc.append(pitch_shifted_mfcc)

    # Create a pitch shifted version of the mfcc, first find the sampling rate
    #pitch_shifted_mfcc = librosa.feature.mfcc(y=pitch_shift(signal, sample_rate, -2), sr=sample_rate, n_mfcc=40)
    pitch_shifted_mfcc = extract_mfcc_1d(pitch_shift(signal, sample_rate, -2), n_mfcc=40, load=False, sample_rate=sample_rate)
    train_set_mfcc.append(pitch_shifted_mfcc)

    # Create a noisy version of the mfcc
    #noisy_mfcc = librosa.feature.mfcc(y=add_background_noise(signal, 0.005), sr=sample_rate, n_mfcc=40)
    noisy_mfcc = extract_mfcc_1d(add_background_noise(signal, 0.005), n_mfcc=40, load=False, sample_rate=sample_rate)
    train_set_mfcc.append(noisy_mfcc)

    # Create a noisy pitch shifted version of the mfcc
    # noisy_pitch_shifted_mfcc = librosa.feature.mfcc(y=add_background_noise(pitch_shift(signal, sample_rate, 2), 0.005), sr=sample_rate, n_mfcc=40)
    noisy_pitch_shifted_mfcc = extract_mfcc_1d(add_background_noise(pitch_shift(signal, sample_rate, 2), 0.005), n_mfcc=40, load=False, sample_rate=sample_rate)
    train_set_mfcc.append(noisy_pitch_shifted_mfcc)

    # Create another noisy pitch shifted version of the mfcc
    #noisy_pitch_shifted_mfcc = librosa.feature.mfcc(y=add_background_noise(pitch_shift(signal, sample_rate, -2), 0.005), sr=sample_rate, n_mfcc=40)
    noisy_pitch_shifted_mfcc = extract_mfcc_1d(add_background_noise(pitch_shift(signal, sample_rate, -2), 0.005), n_mfcc=40, load=False, sample_rate=sample_rate)
    train_set_mfcc.append(noisy_pitch_shifted_mfcc)

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
val_set_mfcc = []
val_set_labels = []
i = 0
for file in val_set:
    # Check if file_name exists, if it doesn't, skip it
    file_name = 'recordings/recordings/' + file + '.mp3'
    if not os.path.exists(file_name):
        continue

    mfcc = extract_mfcc_1d(file_name, n_mfcc=40, load=True)

    val_set_mfcc.append(mfcc)

    # No need to apply data augmentation to the validation set, add correct label
    if 'english' in file:
        val_set_labels.append(1)
    else:
        val_set_labels.append(0)

    # Add a counter to keep track of progress
    i += 1
    if i % 100 == 0:
        print(str(i) + " out of " + str(len(val_set)) + " files processed.")

# Convert the lists to numpy arrays
train_set_mfcc = np.array(train_set_mfcc)
val_set_mfcc = np.array(val_set_mfcc)

# Convert the labels to numpy arrays
train_set_labels = np.array(train_set_labels)
val_set_labels = np.array(val_set_labels)

# Print shapes of first and last mfcc in both lists
print(train_set_mfcc[0].shape)
print(train_set_mfcc[-1].shape)

print(val_set_mfcc[0].shape)
print(val_set_mfcc[-1].shape)

# Print the shape of the arrays
print("Size of training set: ", train_set_mfcc.shape)
print("Size of validation set: ", val_set_mfcc.shape)

# Count the number of english and non-english speakers in the training and validation sets
print("Number of native english speakers in training set: ", np.sum(train_set_labels))
print("Number of non-native english speakers in training set: ", len(train_set_labels) - np.sum(train_set_labels))

print("Number of native english speakers in validation set: ", np.sum(val_set_labels))
print("Number of non-native english speakers in validation set: ", len(val_set_labels) - np.sum(val_set_labels))


# Save the data to npz files
file_name = 'mfcc_data_40_augmented.npz'
print("Saving 40 mfcc coefficients with data augmentation to file: ", file_name)
np.savez(file_name, X_train=train_set_mfcc, y_train=train_set_labels, X_val=val_set_mfcc, y_val=val_set_labels)


