import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam


import os
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split 
from datetime import datetime
import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
#import seaborn as sns
import librosa.display

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd

from plotly.subplots import make_subplots

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile


# directory exploration
train_audio_path = r'/home/agamjeet/for/for-2seconds/training'

dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]

dirs.sort()

print('Number of labels: ' + str(len(dirs)))

# Print the list of directories (labels)
print("List of directories (labels):")
print(dirs)
# Calculate
number_of_recordings = []
for direct in dirs:
    waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]
    number_of_recordings.append(len(waves))
print(number_of_recordings)


#mfcc creation

def process_audio_data(train_audio_path, max_pad_length=86):
    def extract_feature(file_name):
        try:
            audio, sample_rate = librosa.load(file_name)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            pad_width = max_pad_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, max(0, pad_width))), mode='constant')
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            return None
        return mfccs

    # Load and preprocess 'fake' data
    fake_dir = os.path.join(train_audio_path, 'fake')
    data_fake = []
    for file_name in tqdm(os.listdir(fake_dir)):
        file_path = os.path.join(fake_dir, file_name)
        features = extract_feature(file_path)
        if features is not None:
            data_fake.append([features, 0])

    # Load and preprocess 'real' data
    real_dir = os.path.join(train_audio_path, 'real')
    data_real = []
    for file_name in tqdm(os.listdir(real_dir)):
        file_path = os.path.join(real_dir, file_name)
        features = extract_feature(file_path)
        if features is not None:
            data_real.append([features, 1])

    # Concatenate the 'fake' and 'real' data
    data = data_fake + data_real

    # Shuffle the data
    #random.shuffle(data)

    # Split the data and labels
    X_data = np.array([x[0] for x in data])
    y_labels = np.array([x[1] for x in data])

    # Optional: Display the first few rows of the dataset
    dataset_df = pd.DataFrame(data, columns=['feature', 'class_label'])
    print(dataset_df.head())

    # Save data and labels as .npy files with filenames based on train_audio_path
    data_filename = os.path.basename(train_audio_path) + '_data.npy'
    labels_filename = os.path.basename(train_audio_path) + '_labels.npy'
    np.save(data_filename, X_data)
    np.save(labels_filename, y_labels)

    return X_data, y_labels

# Example usage:
train_audio_path = '/home/agamjeet/for/for-2seconds/training'
testing_audio_path = '/home/agamjeet/for/for-2seconds/testing'
validation_audio_path = '/home/agamjeet/for/for-2seconds/validation'

process_audio_data(train_audio_path)
process_audio_data(validation_audio_path)
process_audio_data(testing_audio_path)


