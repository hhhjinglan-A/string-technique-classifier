import sys
import numpy as np
from common import load_audio, preprocess, load_model, predict, SR, NAMES

path = sys.argv[1]
model, mean, std = load_model()
audio = load_audio(path)

hop = SR // 2
windows, times = [], []
for start in range(0, max(len(audio) - SR, 0) + 1, hop):
    w = audio[start:start + SR]
    if np.max(np.abs(w)) < 0.01:
        continue
    windows.append(preprocess(w))
    times.append(start / SR)

if len(windows) == 0:
    print("The recording is too quiet.")
else:
    pred = predict(model, mean, std, windows)
    for t, p in zip(times, pred):
        print(f"{t:5.1f} s  ->  {NAMES[p]}")
        