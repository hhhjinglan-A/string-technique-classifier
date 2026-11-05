import numpy as np
import torch
from torch import nn
import torchaudio
import soundfile as sf

SR = 44100
LENGTH = SR
NAMES = ["arco", "pizz", "tremolo"]

# ---------- audio ----------
def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = torchaudio.functional.resample(
            torch.tensor(audio, dtype=torch.float32), sr, SR).numpy()
    return audio

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

def preprocess(audio):
    return fix_length(trim_silence(normalize(audio)))

# ---------- features ----------
_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=1024, hop_length=512, n_mels=64
)
_to_db = torchaudio.transforms.AmplitudeToDB()

def to_mel(audios):
    X = torch.tensor(np.array(audios), dtype=torch.float32)
    return _to_db(_mel(X)).unsqueeze(1)

# ---------- model ----------
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

def load_model(path="stringnet.pt"):
    saved = torch.load(path)
    model = StringNet()
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved["mean"], saved["std"]

def predict(model, mean, std, audios):
    M = (to_mel(audios) - mean) / std
    with torch.no_grad():
        return model(M).argmax(dim=1)

INSTRUMENTS = ["violin", "viola", "cello", "double_bass"]

class StringNetMT(nn.Module):
    def __init__(self, n_tech=3, n_inst=4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.tech_head = nn.Linear(32, n_tech)
        self.inst_head = nn.Linear(32, n_inst)

    def forward(self, x):
        h = self.body(x)
        return self.tech_head(h), self.inst_head(h)