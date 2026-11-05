import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(0)
names = ["arco", "pizz", "tremolo"]

# ---------- 1. Data ----------
data = np.load("features.npz")
M = torch.tensor(data["M"], dtype=torch.float32).unsqueeze(1)
y = torch.tensor(data["y"], dtype=torch.long)

idx = torch.randperm(len(M))
n_train = int(0.8 * len(M))
train_idx, val_idx = idx[:n_train], idx[n_train:]
X_train, y_train = M[train_idx], y[train_idx]
X_val, y_val = M[val_idx], y[val_idx]

mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
print("Train:", X_train.shape, " Val:", X_val.shape)

# ---------- 2. Model ----------
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

model = StringNet()

# ---------- 3. Training ----------
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def accuracy(X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
    model.train()
    return (pred == y).float().mean().item()

history = {"loss": [], "train": [], "val": []}

for epoch in range(50):
    total_loss = 0
    for xb, yb in train_loader:
        out = model(xb)
        loss = loss_fn(out, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_acc = accuracy(X_train, y_train)
    val_acc = accuracy(X_val, y_val)
    history["loss"].append(avg_loss)
    history["train"].append(train_acc)
    history["val"].append(val_acc)
    print(f"Epoch {epoch+1:2d}  loss {avg_loss:.3f}  "
          f"train acc {train_acc:.2f}  val acc {val_acc:.2f}")

torch.save({"model": model.state_dict(), "mean": mean, "std": std}, "stringnet.pt")
print("Saved stringnet.pt")

# ---------- 4. Evaluation ----------
model.eval()
with torch.no_grad():
    pred = model(X_val).argmax(dim=1)

cm = torch.zeros(3, 3, dtype=torch.long)
for t, p in zip(y_val, pred):
    cm[t, p] += 1

print()
print("Confusion matrix (row = true, column = predicted)")
print(" " * 10, "".join(f"{n:>10}" for n in names))
for i, n in enumerate(names):
    print(f"{n:>10}", "".join(f"{v:>10}" for v in cm[i].tolist()))

# ---------- 5. Figures ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["loss"])
ax1.set_title("Training loss")
ax1.set_xlabel("Epoch")
ax2.plot(history["train"], label="train")
ax2.plot(history["val"], label="validation")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.legend()
plt.tight_layout()
plt.savefig("training_curve.png")

fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3), names)
ax.set_yticks(range(3), names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
for i in range(3):
    for j in range(3):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center")
ax.set_title("Confusion matrix (validation)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved training_curve.png and confusion_matrix.png")
