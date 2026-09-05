import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Motor Diagnostic System", layout="centered")
st.title("Motor / Pump Acoustic Diagnostic System")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

uploaded_file = st.file_uploader("Upload Pump Audio (.wav)", type=["wav"])

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=22050, duration=3.0)
    
    st.audio(uploaded_file)

    st.subheader("Frequency Spectrum (FFT)")
    fft_vals = np.abs(np.fft.rfft(y))
    fft_freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(fft_freqs[:3000], fft_vals[:3000])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Feature extraction
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    feats = np.hstack([centroid, bandwidth, rolloff, zcr, rms, mfccs]).reshape(1, -1)

    # Decision
    pred = model.predict(feats)[0]
    score = model.decision_function(feats)[0]

    st.subheader("Diagnostic Verdict")
    if pred == 1:
        st.success(f"NORMAL OPERATION (Health Score: {score:.3f})")
    else:
        st.error(f"POSSIBLE FAULT DETECTED (Anomaly Score: {score:.3f})")