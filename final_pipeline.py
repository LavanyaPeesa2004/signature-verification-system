import os
import cv2
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from simease_network import SiameseNetwork

# ==================================
# USER INPUT
# ==================================

person_name = input(
    "Enter Person Name (Person1-Person7): "
).strip()

document_path = input(
    "Enter Document Path: "
).strip()

# ==================================
# PATHS
# ==================================

YOLO_MODEL = r"E:\Models\signature\runs\detect\runs\sig\weights\best.pt"

SNN_MODEL = r"E:\Models\signature\best_snn_finetuned.pth"

PERSON_FOLDER = rf"E:\Models\signature\Reference_Images\{person_name}"

THRESHOLD = 65

# ==================================
# CHECKS
# ==================================

if not os.path.exists(document_path):
    print("Document not found!")
    exit()

if not os.path.exists(PERSON_FOLDER):
    print("Person folder not found!")
    exit()

# ==================================
# YOLO DETECTION
# ==================================

print("\nDetecting Signature...")

yolo_model = YOLO(YOLO_MODEL)

img = cv2.imread(document_path)

if img is None:
    print("Failed to load document!")
    exit()

results = yolo_model.predict(
    source=document_path,
    conf=0.25,
    save=False,
    verbose=False
)

best_conf = -1
best_box = None

for result in results:

    if len(result.boxes) == 0:
        continue

    for box in result.boxes:

        conf = float(box.conf[0])

        if conf > best_conf:

            best_conf = conf
            best_box = box

if best_box is None:

    print("No signature detected!")
    exit()

x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()

x1 = int(x1)
y1 = int(y1)
x2 = int(x2)
y2 = int(y2)

signature_crop = img[y1:y2, x1:x2]

crop_path = "signature_crop.png"

cv2.imwrite(
    crop_path,
    signature_crop
)

# ==================================
# LOAD SNN MODEL
# ==================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

snn_model = SiameseNetwork().to(device)

snn_model.load_state_dict(
    torch.load(
        SNN_MODEL,
        map_location=device
    )
)

snn_model.eval()

# ==================================
# TRANSFORM
# ==================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# ==================================
# LOAD TEST SIGNATURE
# ==================================

test_img = Image.open(
    crop_path
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

            emb1, emb2 = snn_model(
                test_img,
                ref_img
            )

            distance = F.pairwise_distance(
                emb1,
                emb2
            )

            similarity = max(
                0,
                100 - (
                    distance.item() * 100
                )
            )

            similarities.append(
                similarity
            )

        except:
            pass

# ==================================
# TOP-3 MATCHING
# ==================================

if len(similarities) == 0:

    print("No reference signatures found!")
    exit()

similarities.sort(
    reverse=True
)

top3 = similarities[:3]

similarity_score = (
    sum(top3) / len(top3)
)

# ==================================
# FINAL RESULT
# ==================================

if similarity_score >= THRESHOLD:

    result = "GENUINE"

else:

    result = "FORGED"

print("\n================================")
print(f"Person: {person_name}")
print(f"Detection Confidence: {best_conf:.2f}")
print(f"Similarity Score: {similarity_score:.2f}%")
print(f"RESULT: {result}")
print("================================")