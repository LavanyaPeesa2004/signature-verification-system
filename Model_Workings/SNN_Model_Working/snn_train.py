import torch
from torch.utils.data import DataLoader

from cindar_dataset_load import CedarPairsDataset
from simease_network import SiameseNetwork
from contrastive_loss import ContrastiveLoss

# =========================
# CONFIG
# =========================

DATASET_PATH = r"E:\Models\signature\bhsigdataset\CEDAR\CEDAR"

BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# =========================
# DATASET
# =========================

print("Creating Dataset...")
dataset = CedarPairsDataset(DATASET_PATH)

print("Dataset Created!")
print("Dataset Size:", len(dataset))

print("Creating DataLoader...")
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("DataLoader Ready!")

# =========================
# MODEL
# =========================

print("Loading Model...")

model = SiameseNetwork().to(DEVICE)

criterion = ContrastiveLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

print("Model Ready!")

# =========================
# TRAINING
# =========================

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for img1, img2, labels in train_loader:

        img1 = img1.to(DEVICE)
        img2 = img2.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        emb1, emb2 = model(img1, img2)

        loss = criterion(
            emb1,
            emb2,
            labels
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f}"
    )

    # Save every epoch
    torch.save(
        model.state_dict(),
        f"snn_epoch_{epoch+1}.pth"
    )

    print(
        f"Saved: snn_epoch_{epoch+1}.pth"
    )

    # Save best model
    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            model.state_dict(),
            "best_snn_model.pth"
        )

        print(
            "Best model updated!"
        )

print("\nTraining Complete!")
print(
    f"Best Loss: {best_loss:.6f}"
)