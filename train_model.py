import numpy as np
import librosa
from sklearn.ensemble import IsolationForest
import pickle

def extract_features(signal, sr=22050):
    centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))
    rms = np.mean(librosa.feature.rms(y=signal))
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13), axis=1)
    return np.hstack([centroid, bandwidth, rolloff, zcr, rms, mfccs])

# Generate baseline training data (clean 50Hz motor hum + slight noise)
sr = 22050
t = np.linspace(0, 3, sr * 3)
X_train = []

for _ in range(50):
    noise = np.random.normal(0, 0.05, len(t))
    clean_pump = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 100 * t) + noise
    feats = extract_features(clean_pump.astype(np.float32), sr=sr)
    X_train.append(feats)

model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("SUCCESS: model.pkl created!")