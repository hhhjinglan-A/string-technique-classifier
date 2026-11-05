from pathlib import Path
from collections import Counter

instruments = ["cello", "violin", "viola", "double_bass"]
all_techniques = []

for inst in instruments:
    files = Path("data", inst).glob("*.mp3")
    for f in files:
        all_techniques.append(f.stem.split("_")[-1])

counts = Counter(all_techniques)
print("Total files:", len(all_techniques))

for name, n in counts.most_common(20):
    print(n, name)
    