import queue
import numpy as np
import sounddevice as sd
from common import preprocess, load_model, predict, SR, NAMES

model, mean, std = load_model()

# ---------- 1. measure room noise ----------
print("Measuring room noise... please be quiet for 2 seconds.")
noise = sd.rec(2 * SR, samplerate=SR, channels=1)
sd.wait()
noise_level = np.max(np.abs(noise))
THRESHOLD = noise_level * 2
print(f"Room noise = {noise_level:.3f}, threshold = {THRESHOLD:.3f}")

# ---------- 2. listen ----------
buffer = np.zeros(SR, dtype=np.float32)
q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(indata[:, 0].copy())

print("Listening... press Control + C to stop.")
try:
    with sd.InputStream(samplerate=SR, channels=1, blocksize=SR // 2, callback=callback):
        while True:
            block = q.get()
            buffer = np.concatenate([buffer[len(block):], block])
            level = np.max(np.abs(buffer))
            if level < THRESHOLD:
                print(f"{level:.3f}  (silence)")
                continue
            pred = predict(model, mean, std, [preprocess(buffer)])
            print(f"{level:.3f}  {NAMES[pred[0]]}")
except KeyboardInterrupt:
    print("Stopped.")
    