import re
import matplotlib.pyplot as plt

path = "../../checkpoints/checkpoint_results.yaml"

epochs = []
train_loss = []
val_loss = []
pix_acc = []
miou = []
lr = []

with open(path, "r", encoding="utf-8") as f:
    content = f.read().split("----------------------------------------")

for block in content:
    block = block.strip()
    if not block:
        continue

    m_epoch = re.search(r"Epoch:\s*(\d+)", block)
    m_lr = re.search(r"Learning Rate:\s*([0-9.]+)", block)
    m_tr = re.search(r"Training Loss:\s*([0-9.]+)", block)
    m_vl = re.search(r"Validation Loss:\s*([0-9.]+)", block)
    m_pa = re.search(r"Pixel Accuracy:\s*([0-9.]+)", block)
    m_mi = re.search(r"Mean IoU:\s*([0-9.]+)", block)

    if not (m_epoch and m_lr and m_tr and m_vl and m_pa and m_mi):
        continue

    epochs.append(int(m_epoch.group(1)))
    lr.append(float(m_lr.group(1)))
    train_loss.append(float(m_tr.group(1)))
    val_loss.append(float(m_vl.group(1)))
    pix_acc.append(float(m_pa.group(1)))
    miou.append(float(m_mi.group(1)))

# Loss curves
plt.figure(figsize=(7,4))
plt.plot(epochs, train_loss, label="Train loss")
plt.plot(epochs, val_loss, label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=200)

# mIoU curve
plt.figure(figsize=(7,4))
plt.plot(epochs, miou, label="Val mIoU")
plt.xlabel("Epoch")
plt.ylabel("mIoU")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("miou_curve.png", dpi=200)

# LR schedule
plt.figure(figsize=(7,4))
plt.plot(epochs, lr, label="Learning rate")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("lr_curve.png", dpi=200)