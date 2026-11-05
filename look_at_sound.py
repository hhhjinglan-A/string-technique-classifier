import soundfile as sf
from pathlib import Path

f = sorted(Path("data/violin").glob("*pizz-normal.mp3"))[0]
audio, sr = sf.read(f)

print("File:", f.name)
print("Sample rate:", sr)
print("Number of samples:", len(audio))
print("Length (seconds):", len(audio) / sr)
print("Shape:", audio.shape)
print("First 10 numbers:", audio[:10])
