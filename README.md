# Synthetic_voice_detection
Dataset used -https://bil.eecs.yorku.ca/datasets/ 
The Fake-or-Real Dataset

The Fake-or-Real (FoR) dataset is a collection of more than 195,000 utterances from real humans and computer generated speech. The dataset can be used to train classifiers to detect synthetic speech.
The dataset aggregates data from the latest TTS solutions (such as Deep Voice 3 and Google Wavenet TTS) as well as a variety of real human speech, including the Arctic Dataset (http://festvox.org/cmu_arctic/), LJSpeech Dataset (https://keithito.com/LJ-Speech-Dataset/), VoxForge Dataset (http://www.voxforge.org) and our own speech recordings.
The dataset is published in four versions: for-original, for-norm, for-2sec and for-rerec.
The third one , named for-2sec is based on the second one, but with the files truncated at 2 seconds.This subset has been used
To save time for this project only the testing part of the dataset has been split into testing ,training and validation 


The audio files are preprocessed by extracting features using the Mel-frequency cepstral coefficients (MFCCs) method. MFCCs are widely used in audio signal processing due to their ability to capture essential characteristics of audio signals.

The CNN model consists of a total of 17 layers, including custom Causal Gated Conv1D layers, Concatenate layers, Add layers, Conv1D layers, GlobalAveragePooling1D layer, and Activation layer. It is designed to effectively process sequential audio data and make accurate predictions for audio classification tasks. The architecture can be adjusted by controlling the number of filters and the width of the network using the width_multiply parameter.

After training the model on the test set, its performance was evaluated using two key metrics: test loss and test accuracy.

The test loss, which measures the dissimilarity between the model's predictions and the actual labels, was found to be approximately 0.1253. Lower test loss values suggest better agreement between predictions and true labels.

The test accuracy, expressed as a percentage, stands at around 96.17%. This indicates that the model correctly classified approximately 96.17% of the instances in the test set.
