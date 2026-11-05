import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from common import to_mel, StringNetMT, NAMES, INSTRUMENTS

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---------- 1. Data ----------
data = np.load("dataset_v4.npz")
tech = torch.tensor(data["tech"])
inst = torch.tensor(data["inst"])
group = data["group"]
X = data["X"]
M = torch.cat([to_mel(X[i:i + 100]) for i in range(0, len(X), 100)])

# split by group: the same note never appears in both train and test
groups = np.unique(group)
rng.shuffle(groups)
train_groups = set(groups[:int(0.8 * len(groups))])
is_train = torch.tensor([g in train_groups for g in group])

X_train, X_test = M[is_train], M[~is_train]
t_train, t_test = tech[is_train], tech[~is_train]
i_train, i_test = inst[is_train], inst[~is_train]

mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
print("Train:", len(X_train), " Test:", len(X_test))

# ---------- 2. Model and training ----------
model = StringNetMT()
loader = DataLoader(TensorDataset(X_train, t_train, i_train), batch_size=32, shuffle=True)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def evaluate(X):
    model.eval()
    with torch.no_grad():
        out_t, out_i = model(X)
    model.train()
    return out_t.argmax(dim=1), out_i.argmax(dim=1)

for epoch in range(40):
    total_loss = 0
    for xb, tb, ib in loader:
        out_t, out_i = model(xb)
        loss = loss_fn(out_t, tb) + loss_fn(out_i, ib)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    pt, pi = evaluate(X_test)
    print(f"Epoch {epoch+1:2d}  loss {total_loss / len(loader):.3f}  "
          f"test technique {(pt == t_test).float().mean():.2f}  "
          f"test instrument {(pi == i_test).float().mean():.2f}")

torch.save({"model": model.state_dict(), "mean": mean, "std": std}, "stringnet_v4.pt")
print("Saved stringnet_v4.pt")

# ---------- 3. Confusion matrices ----------
def confusion(true, pred, n):
    cm = torch.zeros(n, n, dtype=torch.long)
    for a, b in zip(true, pred):
        cm[a, b] += 1
    return cm

pt, pi = evaluate(X_test)
cm_t = confusion(t_test, pt, 3)
cm_i = confusion(i_test, pi, 4)
print("\nTechnique (row = true, column = predicted)")
print(cm_t)
print("\nInstrument (row = true, column = predicted)")
print(cm_i)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm, labels, title in [(axes[0], cm_t, NAMES, "Technique"),
                              (axes[1], cm_i, INSTRUMENTS, "Instrument")]:
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + " (test set)")
    for r in range(len(labels)):
        for c in range(len(labels)):
            ax.text(c, r, int(cm[r, c]), ha="center", va="center")
plt.tight_layout()
plt.savefig("confusion_v4.png")
print("Saved confusion_v4.png")