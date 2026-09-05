import numpy as np
import soundfile as sf

sr = 22050
duration = 3.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# 1. NORMAL MOTOR: Clean 50 Hz hum + slight noise
normal_signal = 0.6 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 100 * t)
normal_noise = np.random.normal(0, 0.03, len(t))
normal_audio = (normal_signal + normal_noise).astype(np.float32)
sf.write("sample_normal.wav", normal_audio, sr)

# 2. FAULTY PUMP: High-frequency friction / cavitation squeal
fault_harmonic = 0.3 * np.sin(2 * np.pi * 50 * t)
cavitation_noise = np.random.normal(0, 0.25, len(t)) * np.sin(2 * np.pi * 2800 * t)
bearing_impacts = 0.4 * np.sin(2 * np.pi * 1500 * t) * (np.sin(2 * np.pi * 12 * t) > 0.8)
fault_audio = (fault_harmonic + cavitation_noise + bearing_impacts).astype(np.float32)
sf.write("sample_fault.wav", fault_audio, sr)

print("Created sample_normal.wav and sample_fault.wav successfully!")