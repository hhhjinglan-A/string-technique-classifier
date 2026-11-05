import random
import re
import numpy as np
from pathlib import Path
from common import load_audio, preprocess, INSTRUMENTS

random.seed(0)

ORCH = Path("/Volumes/HJL/datasets/_OrchideaSOL2020_release/OrchideaSOL2020/Strings")
PHIL_FOLDER = {"violin": "violin", "viola": "viola", "cello": "cello", "double_bass": "double_bass"}
ORCH_FOLDER = {"violin": "Violin", "viola": "Viola", "cello": "Violoncello", "double_bass": "Contrabass"}
PHIL_TECH = ["arco-normal", "pizz-normal", "arco-tremolo"]
ORCH_TECH = ["ordinario", "pizzicato_l_vib", "tremolo"]

def find_pitch(name):
    for part in re.split(r"[_-]", name):
        if re.fullmatch(r"[A-G](#|s)?[0-9]", part):
            return part
    return "unknown"

def pick(files, n):
    files = sorted(files)
    if len(files) > n:
        return random.sample(files, n)
    return files

items = []
for i, inst in enumerate(INSTRUMENTS):
    for t in range(3):
        # Philharmonia (long arco only, like v2/v3)
        files = list(Path("data", PHIL_FOLDER[inst]).glob(f"*_{PHIL_TECH[t]}.mp3"))
        if t == 0:
            files = [f for f in files if f.stem.split("_")[2] not in ("025", "05")]
        for f in pick(files, 30):
            items.append((f, t, i, f"phil-{inst}-{find_pitch(f.stem)}"))
        # OrchideaSOL (all four instruments, including cello)
        files = (ORCH / ORCH_FOLDER[inst] / ORCH_TECH[t]).glob("*.wav")
        for f in pick(files, 60):
            items.append((f, t, i, f"orch-{inst}-{find_pitch(f.stem)}"))

print("Total files:", len(items))

X, tech, inst, group = [], [], [], []
for f, t, i, g in items:
    try:
        a = load_audio(f)
    except Exception:
        print("Skip (cannot read):", f.name)
        continue
    if np.max(np.abs(a)) == 0:
        continue
    X.append(preprocess(a))
    tech.append(t)
    inst.append(i)
    group.append(g)

X = np.array(X, dtype=np.float32)
tech = np.array(tech)
inst = np.array(inst)
group = np.array(group)
print("X shape:", X.shape)
print("technique counts (arco, pizz, tremolo):", np.bincount(tech))
print("instrument counts (violin, viola, cello, bass):", np.bincount(inst))
print("number of groups:", len(np.unique(group)))
np.savez("dataset_v4.npz", X=X, tech=tech, inst=inst, group=group)
print("Saved dataset_v4.npz")
