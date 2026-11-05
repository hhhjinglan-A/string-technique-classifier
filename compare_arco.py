import numpy as np
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

ORCH = Path("/Volumes/HJL/datasets/_OrchideaSOL2020_release/OrchideaSOL2020/Strings/Violoncello")
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

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=1024, hop_length=512, n_mels=64
)
to_db = torchaudio.transforms.AmplitudeToDB()

examples = [
    ("1. Philharmonia arco 0.25 s", sorted(Path("data/cello").glob("*_025_*_arco-normal.mp3"))[0]),
    ("2. Philharmonia arco 1.5 s",  sorted(Path("data/cello").glob("*_15_*_arco-normal.mp3"))[0]),
    ("3. Philharmonia tremolo",     sorted(Path("data/cello").glob("*_arco-tremolo.mp3"))[0]),
    ("4. OrchideaSOL ordinario",    sorted((ORCH / "ordinario").glob("*.wav"))[0]),
    ("5. OrchideaSOL tremolo",      sorted((ORCH / "tremolo").glob("*.wav"))[0]),
]

fig, axes = plt.subplots(2, 5, figsize=(22, 7))

for col, (title, f) in enumerate(examples):
    audio, sr = sf.read(f)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = torchaudio.functional.resample(
            torch.tensor(audio, dtype=torch.float32), sr, SR).numpy()
    audio = fix_length(trim_silence(normalize(audio)))

    t = np.arange(LENGTH) / SR
    axes[0, col].plot(t, audio)
    axes[0, col].set_ylim(-1.05, 1.05)
    axes[0, col].set_title(title)

    M = to_db(mel(torch.tensor(audio, dtype=torch.float32)))
    axes[1, col].imshow(M.numpy(), origin="lower", aspect="auto")
    axes[1, col].set_title(f.name, fontsize=8)

plt.tight_layout()
plt.savefig("compare_arco.png")
print("Saved compare_arco.png")
