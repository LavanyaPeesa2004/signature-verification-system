import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from simease_network import SiameseNetwork

# ==================================
# CONFIG
# ==================================

MODEL_PATH = "best_snn_finetuned.pth"

PERSON_FOLDER = r"E:\Models\signature\Reference_Images\Person3"

TEST_IMAGE = r"E:\Models\signature\testsample4.jpeg"

THRESHOLD = 65

# ==================================
# DEVICE
# ==================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ==================================
# TRANSFORM
# ==================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# ==================================
# LOAD MODEL
# ==================================

model = SiameseNetwork().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model Loaded!")

# ==================================
# LOAD TEST IMAGE
# ==================================

test_img = Image.open(
    TEST_IMAGE
).convert("RGB")

test_img = transform(
    test_img
).unsqueeze(0).to(device)

# ==================================
# COMPARE
# ==================================

similarities = []

with torch.no_grad():

    for img_name in os.listdir(PERSON_FOLDER):

        img_path = os.path.join(
            PERSON_FOLDER,
            img_name
        )

        try:

            ref_img = Image.open(
                img_path
            ).convert("RGB")

            ref_img = transform(
                ref_img
            ).unsqueeze(0).to(device)

            emb1, emb2 = model(
                test_img,
                ref_img
            )

            distance = F.pairwise_distance(
                emb1,
                emb2
            )

            similarity = max(
                0,
                100 - (distance.item() * 100)
            )

            similarities.append(
                similarity
            )

            print(
                f"{img_name} -> {similarity:.2f}%"
            )

        except Exception as e:

            print(
                f"Skipped: {img_name}"
            )

# ==================================
# FINAL RESULT
# ==================================

if len(similarities) == 0:

    print("No reference signatures found!")
    exit()

# Sort highest similarity first
similarities.sort(reverse=True)

# Take Top 3 matches
top_k = similarities[:3]

avg_similarity = (
    sum(top_k) / len(top_k)
)

print("\n====================")
print("TOP 3 MATCHES:")

for i, score in enumerate(top_k, start=1):

    print(
        f"Top {i}: {score:.2f}%"
    )

print("\n====================")
print(
    f"Top-3 Average Similarity: {avg_similarity:.2f}%"
)

if avg_similarity >= THRESHOLD:

    print("RESULT: GENUINE")

else:

    print("RESULT: FORGED")