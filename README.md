# AI-Based Signature Verification System

## Overview

This project presents an AI-powered Signature Verification System that automatically detects, extracts, and verifies handwritten signatures from document images.

The system combines YOLO (You Only Look Once) object detection for signature localization and a Siamese Neural Network (SNN) for signature verification. Given a document and a claimed identity, the system determines whether the signature is genuine or forged.

---

## Features

* Automatic signature detection using YOLO
* Signature extraction from document images
* Signature verification using Siamese Neural Networks
* Genuine vs Forged classification
* Top-3 similarity matching strategy for improved robustness
* User-selectable identity verification
* End-to-end automated verification pipeline

---

## System Workflow

Document Image
→ Signature Detection (YOLO)
→ Signature Cropping
→ Feature Extraction (Siamese Network)
→ Similarity Computation
→ Genuine / Forged Decision

---

## Technologies Used

* Python
* PyTorch
* OpenCV
* Ultralytics YOLO
* Siamese Neural Network (ResNet18 Backbone)
* NumPy
* PIL (Python Imaging Library)

---

## Dataset

### YOLO Training Dataset

Custom annotated signature dataset containing document images with signature bounding boxes.

### Signature Verification Dataset

* CEDAR Signature Dataset
* Custom signature reference dataset

The Siamese Network was fine-tuned using custom positive and negative signature pairs.

---

## Model Architecture

### Signature Detection

YOLO Object Detection Model

* Detects signature locations in document images
* Generates bounding box coordinates
* Extracts signature region automatically

### Signature Verification

Siamese Neural Network

* ResNet18 feature extractor
* 128-dimensional signature embeddings
* Contrastive Loss for similarity learning

Positive Pairs:

* Same person's signatures

Negative Pairs:

* Different persons' signatures

---

## Project Structure

```text
signature-verification-system/

├── final_pipeline.py
├── verify_signature.py
├── simease_network.py
├── contrastive_loss.py
├── custom_dataset_load.py
├── custom_pair_generator.py
├── snn_finetune.py
├── signature_train_yolo26.py
├── data.yaml
├── README.md
├── Reference_Images/
├── Yolo_Dataset/
├── Simease_Dataset/
└── runs/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/LavanyaPeesa2004/signature-verification-system.git

cd signature-verification-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the final verification pipeline:

```bash
python final_pipeline.py
```

Enter:

```text
Person Name (Person1-Person7)
Document Image Path
```

Example:

```text
Enter Person Name: Person3

Enter Document Path:
E:\Models\signature\test_samples\309.jpg
```

Output:

```text
================================

Person: Person3

Detection Confidence: 0.86

Similarity Score: 78.84%

RESULT: GENUINE

================================
```

---

## Results

The system successfully:

* Detects signatures from document images
* Extracts signature regions automatically
* Verifies signature authenticity
* Distinguishes genuine and forged signatures using similarity-based verification

---

## Future Enhancements

* Streamlit Web Application
* Multi-user signature database
* Real-time signature verification
* Cloud deployment
* Improved signature preprocessing
* Enhanced verification accuracy using larger datasets

---

## Author

**Lavanya Peesa**

Data Science Engineering Student

GitHub:
https://github.com/LavanyaPeesa2004

---

## License

This project is developed for educational and research purposes.
