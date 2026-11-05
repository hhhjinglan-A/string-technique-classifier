import random
import numpy as np
from pathlib import Path
from common import load_audio, preprocess

random.seed(0)

ORCH = Path("/Volumes/HJL/datasets/_OrchideaSOL2020_release/OrchideaSOL2020/Strings")
PHIL_INSTRUMENTS = ["cello", "violin", "viola", "double_bass"]
ORCH_INSTRUMENTS = ["Violin", "Viola", "Contrabass"]   # no Violoncello: cello is the test

phil = {0: "arco-normal", 1: "pizz-normal", 2: "arco-tremolo"}
orch = {0: "ordinario", 1: "pizzicato_l_vib", 2: "tremolo"}

def pick(files, n):
    files = sorted(files)
    if len(files) > n:
        return random.sample(files, n)
    return files

items = []
for label in range(3):
    # Philharmonia (same rule as v2)
    files = []
    for inst in PHIL_INSTRUMENTS:
        files += list(Path("data", inst).glob(f"*_{phil[label]}.mp3"))
    if label == 0:
        files = [f for f in files if f.stem.split("_")[2] not in ("025", "05")]
    files = pick(files, 105)
    print("Philharmonia", phil[label], len(files))
    items += [(f, label) for f in files]

    # OrchideaSOL (NO cello)
    for inst in ORCH_INSTRUMENTS:
        files = pick((ORCH / inst / orch[label]).glob("*.wav"), 60)
        print("OrchideaSOL", inst, orch[label], len(files))
        items += [(f, label) for f in files]

X, y = [], []
for f, label in items:
    try:
        a = load_audio(f)
    except Exception:
        print("Skip (cannot read):", f.name)
        continue
    if np.max(np.abs(a)) == 0:
        continue
    X.append(preprocess(a))
    y.append(label)

X = np.array(X)
y = np.array(y)
print("X shape:", X.shape)
print("count per class (arco, pizz, tremolo):", np.bincount(y))
np.savez("dataset.npz", X=X, y=y)
print("Saved dataset.npz")