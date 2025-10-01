import librosa
import numpy as np

def extract_features(audio_path, sr=16000):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)

    # Temporal features
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    duration = float(librosa.get_duration(y=y, sr=sr))

    # Spectral features
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # MFCC + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)

    mfcc_means = [float(np.mean(mfcc[i])) for i in range(13)]
    delta_mfcc_means = [float(np.mean(delta_mfcc[i])) for i in range(13)]

    # Combine features
    features = [
        zcr, rms, duration,
        spectral_centroid, spectral_bandwidth,
        spectral_contrast, spectral_flatness,
    ] + mfcc_means + delta_mfcc_means

    return np.array(features)


# --- Data Extraction ---
import os
import pandas as pd

dataset_path = "dataset_16k"
features_list = []

os.makedirs("features", exist_ok=True)
csv_path = "features/extracted_features_selected.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded existing features from {csv_path}")
else:
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if not file.endswith(".wav"):
                continue
            file_path = os.path.join(root, file)
            features = extract_features(file_path)
            emotion = int(file.split("-")[2])  # extract emotion ID
            features_list.append(np.append(features, emotion))

    # Columns
    columns = [
        "zcr_mean", "rms_mean", "duration_s",
        "spectral_centroid_mean", "spectral_bandwidth_mean",
        "spectral_contrast_mean", "spectral_flatness_mean"
    ] + [f"mfcc{i+1}" for i in range(13)] \
    + [f"delta_mfcc{i+1}" for i in range(13)] \
    + ["emotion"]

    df = pd.DataFrame(features_list, columns=columns)
    df.to_csv(csv_path, index=False) 
    print(f"Full features saved to {csv_path}")