import librosa
import numpy as np

def extract_features(audio, sr):

    features = []

    # 40 MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(np.mean(centroid))

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features.append(np.mean(bandwidth))

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features.extend(np.mean(contrast, axis=1))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(np.mean(zcr))

    # RMS energy
    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    return np.array(features)
