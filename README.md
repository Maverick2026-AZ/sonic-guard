# SonicGuard 🛡️🔊
**Acoustic Predictive Maintenance & Anomaly Detection Pipeline**

SonicGuard is an edge-ready acoustic diagnostic system that uses microphone audio from rotating machinery (pumps, industrial blowers, motors) to detect mechanical anomalies before critical failures occur.

---

## Architecture Pipeline

```text
                     INDUSTRIAL ROTATING ASSET
                     (Pumps, Motors, Blowers)
                                │
                 Acoustic Noise / Vibrations
       (Friction, Cavitation, Blade Pass, Harmonics)
                                │
                                ▼
                    ┌─────────────────────────┐
                    │     Smartphone MEMS     │
                    │       Microphone        │
                    └────────────┬────────────┘
                                 │
                   Raw 1D Time-Domain Signal
                     x(t) @ 22.05 / 44.1 kHz
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Signal Conditioning   │
                    │  & Preprocessing Stage  │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
  High-Pass Filter       Windowing Segment       Downsampling /
   (fc ~ 150-200 Hz:     (Hanning / Hamming:     Normalizing
  Cut Ambient Rumble)    Minimize Leakage)       Gain Levels
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      FFT Transform      │
                    │    (Time -> Frequency)  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Feature Engineering   │
                    │  (Centroid, MFCC, RMS)  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  SonicGuard ML Engine   │
                    │   (Isolation Forest)    │
                    └────────────┬────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
         [ Score >= 0 ]                    [ Score < 0 ]
        NORMAL OPERATION                  POSSIBLE FAULT
```

---

## Features
- **FFT Spectral Decomposition:** Converts time-domain acoustic signals to inspect harmonic spikes and cavitation sidebands.
- **Unsupervised Anomaly Scoring:** Uses an `IsolationForest` trained on baseline sound envelopes to catch rare faults without requiring labeled failure datasets.
- **Interactive Web UI:** Real-time Streamlit dashboard supporting file upload and spectral visualization.

---

## Setup & Running Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Maverick2026-AZ/sonic-guard.git](https://github.com/Maverick2026-AZ/sonic-guard.git)
   cd sonic-guard
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the baseline model:**
   ```bash
   python train_model.py
   ```

5. **Launch the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```