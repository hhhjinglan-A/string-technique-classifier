import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt

data = np.load("dataset.npz")
X = torch.tensor(data["X"], dtype=torch.float32)
y = data["y"]

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100, n_fft=1024, hop_length=512, n_mels=64
)
to_db = torchaudio.transforms.AmplitudeToDB()

M = to_db(mel(X))
print("Mel shape:", M.shape)

np.savez("features.npz", M=M.numpy(), y=y)
print("Saved features.npz")

names = ["arco-normal", "pizz-normal", "arco-tremolo"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for label in range(3):
    i = np.where(y == label)[0][0]
    axes[label].imshow(M[i].numpy(), origin="lower", aspect="auto")
    axes[label].set_title(names[label])
plt.tight_layout()
plt.savefig("mel_examples.png")
print("Saved mel_examples.png")
