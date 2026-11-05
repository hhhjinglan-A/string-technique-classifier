from pathlib import Path
from collections import defaultdict
import numpy as np
from common import load_audio, preprocess, load_model, predict

model, mean, std = load_model()

groups = defaultdict(list)
for inst in ["cello", "violin", "viola", "double_bass"]:
    for f in Path("data", inst).glob("*_arco-normal.mp3"):
        duration = f.stem.split("_")[2]
        groups[duration].append(f)

for duration in sorted(groups):
    audios = []
    for f in groups[duration]:
        try:
            a = load_audio(f)
        except Exception:
            print("Skip (cannot read):", f.name)
            continue
        if np.max(np.abs(a)) == 0:
            continue
        audios.append(preprocess(a))
    pred = predict(model, mean, std, audios)
    counts = [(pred == k).sum().item() for k in range(3)]
    print(f"{duration:10s} n={len(pred):4d}  "
          f"guessed arco/pizz/tremolo = {counts}  "
          f"arco correct = {counts[0] / len(pred):.2f}")
    