import numpy as np
import matplotlib.pyplot as plt

data = np.load("dataset.npz")
X = data["X"]
y = data["y"]
names = ["arco-normal", "pizz-normal", "arco-tremolo"]
SR = 44100

fig, axes = plt.subplots(3, 3, figsize=(15, 9))

for label in range(3):
    i = np.where(y == label)[0][0]
    sound = X[i]
    t = np.arange(len(sound)) / SR

    axes[label, 0].plot(t, sound)
    axes[label, 0].set_title(names[label] + ": waveform (1 s)")

    start = np.argmax(np.abs(sound))
    zoom = sound[start:start + 882]
    axes[label, 1].plot(zoom)
    axes[label, 1].set_title(names[label] + ": zoom in (0.02 s)")

    axes[label, 2].specgram(sound, Fs=SR, NFFT=1024)
    axes[label, 2].set_ylim(0, 5000)
    axes[label, 2].set_title(names[label] + ": spectrogram")

plt.tight_layout()
plt.savefig("three_techniques.png")
print("Saved three_techniques.png")
