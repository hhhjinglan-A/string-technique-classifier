import numpy as np
import torch
from torch import nn
import torchaudio
import soundfile as sf
from pathlib import Path

ROOT = Path("/Volumes/HJL/datasets/_OrchideaSOL2020_release/OrchideaSOL2020/Strings/Violoncello")
SR = 44100
LENGTH = SR

tests = [
    ("ordinario", 0),
    ("pizzicato_l_vib", 1),
    ("pizzicato_secco", 1),
    ("tremolo", 2),
]

# ----- same preprocessing as prepare_data.py -----
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

# ----- same model as train.py -----
class StringNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.layers(x)

saved = torch.load("stringnet.pt")
model = StringNet()
model.load_state_dict(saved["model"])
model.eval()
mean, std = saved["mean"], saved["std"]

# ----- the test -----
for folder, label in tests:
    files = sorted((ROOT / folder).glob("*.wav"))
    X = []
    for f in files:
        audio, sr = sf.read(f)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != SR:
            audio = torchaudio.functional.resample(
                torch.tensor(audio, dtype=torch.float32), sr, SR).numpy()
        if np.max(np.abs(audio)) == 0:
            continue
        X.append(fix_length(trim_silence(normalize(audio))))

    X = torch.tensor(np.array(X), dtype=torch.float32)
    M = (to_db(mel(X)).unsqueeze(1) - mean) / std
    with torch.no_grad():
        pred = model(M).argmax(dim=1)

    counts = [(pred == k).sum().item() for k in range(3)]
    acc = counts[label] / len(pred)
    print(f"{folder:18s} n={len(pred):4d}  "
          f"guessed arco/pizz/tremolo = {counts}  accuracy = {acc:.2f}")
    