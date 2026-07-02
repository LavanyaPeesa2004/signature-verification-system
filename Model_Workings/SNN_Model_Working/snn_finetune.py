import torch
from torch.utils.data import DataLoader

from custom_dataset_load import CustomPairsDataset
from simease_network import SiameseNetwork
from contrastive_loss import ContrastiveLoss

# ==================================
# CONFIG
# ==================================

PAIR_FILE = "custom_pairs.pkl"

PRETRAINED_MODEL = "best_snn_finetuned.pth"

BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# ==================================
# DATASET
# ==================================

print("Loading Dataset...")

dataset = CustomPairsDataset(
    PAIR_FILE
)

print(
    "Dataset Size:",
    len(dataset)
)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("DataLoader Ready!")

# ==================================
# MODEL
# ==================================

print("Loading Model...")

model = SiameseNetwork().to(DEVICE)

model.load_state_dict(
    torch.load(
        PRETRAINED_MODEL,
        map_location=DEVICE
    )
)

print(
    "Loaded:",
    PRETRAINED_MODEL
)

criterion = ContrastiveLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# ==================================
# TRAINING
# ==================================

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for img1, img2, labels in train_loader:

        img1 = img1.to(DEVICE)
        img2 = img2.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        emb1, emb2 = model(
            img1,
            img2
        )

        loss = criterion(
            emb1,
            emb2,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.6f}"
    )

    # Save every epoch

    torch.save(
        model.state_dict(),
        f"finetune_epoch_{epoch+1}.pth"
    )

    # Save best model

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            model.state_dict(),
            "best_snn_finetuned.pth"
        )

        print(
            "Best model updated!"
        )

print("\nTraining Complete!")

print(
    f"Best Loss: {best_loss:.6f}"
)