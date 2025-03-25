#convert_image_to_csv(img_files)

#convert_audio_to_spectogram(audio_files)


import os
import numpy as np
import pandas as pd
import librosa
import cv2

def convert_images_to_array(folder_fp, img_size=(128, 128), grayscale=True):
    """
    Converts images in a folder to a NumPy array.

    Args:
        folder_fp (str): Folder path containing images.
        img_size (tuple): Target size for resizing images.
        grayscale (bool): Whether to convert images to grayscale.

    Returns:
        np.ndarray: 2D NumPy array where each row represents an image.
    """
    image_list = []
    for filename in os.listdir(folder_fp):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_fp, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
            img = cv2.resize(img, img_size)  # Resize to uniform dimensions
            image_list.append(img.flatten())  # Flatten to 1D and add to list

    return np.array(image_list, dtype=np.float32)  # Return as NumPy array




def convert_audio_to_spectrogram(folder_fp, sr=22050, n_mfcc=20):
    """
    Converts audio files in a folder to MFCC-based NumPy array.

    Args:
        folder_fp (str): Folder path containing audio files.
        sr (int): Sample rate for loading audio.
        n_mfcc (int): Number of MFCC features.

    Returns:
        np.ndarray: 2D NumPy array where each row is an audio file's MFCC.
    """
    audio_list = []
    for filename in os.listdir(folder_fp):
        if filename.lower().endswith((".wav", ".mp3", ".flac")):
            audio_path = os.path.join(folder_fp, filename)
            y, sr = librosa.load(audio_path, sr=sr)  # Load audio
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # Extract MFCC
            audio_list.append(mfcc.mean(axis=1))  # Take mean across time axis

    return np.array(audio_list, dtype=np.float32)  # Return as NumPy array
