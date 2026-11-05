import random
import numpy as np
import soundfile as sf
from pathlib import Path

random.seed(0)

instruments = ["cello", "violin", "viola", "double_bass"]
techniques = ["arco-normal", "pizz-normal", "arco-tremolo"]
SR = 44100
LENGTH = SR

def normalize(audio):
    return audio / np.max(np.abs(audio))

def trim_silence(audio):
    loud = np.abs(audio) > 0.05
    start = np.argmax(loud)
    start = max(start - 441, 0)
    return audio[start:]

def fix_length(audio):
    if len(audio) >= LENGTH:
        return audio[:LENGTH]
    padding = np.zeros(LENGTH - len(audio))
    return np.concatenate([audio, padding])

X = []
y = []

for label, tech in enumerate(techniques):
    files = []
    for inst in instruments:
        files += list(Path("data", inst).glob(f"*_{tech}.mp3"))
    if tech == "arco-normal":
        files = [f for f in files if f.stem.split("_")[2] not in ("025", "05")]
    if len(files) > 105:
        files = random.sample(files, 105)
    print(tech, "->", len(files), "files, label =", label)

    for f in files:
        try:
            audio, sr = sf.read(f)
        except Exception:
            print("Skip (cannot read):", f.name)
            continue
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != SR:
            print("Skip (different sample rate):", f.name, sr)
            continue
        if np.max(np.abs(audio)) == 0:
            print("Skip (silent file):", f.name)
            continue
        audio = normalize(audio)
        audio = trim_silence(audio)
        X.append(fix_length(audio))
        y.append(label)

X = np.array(X)
y = np.array(y)
print("X shape:", X.shape)
print("y shape:", y.shape)
np.savez("dataset.npz", X=X, y=y)
print("Saved dataset.npz")
