import cv2
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from paddleocr import PaddleOCR

# =========================
# LOAD PADDLE OCR
# =========================

print("Loading PaddleOCR...")
ocr = PaddleOCR(lang='en')
print("PaddleOCR loaded.")

# =========================
# LOAD CNN LANGUAGE CLASSIFIER
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_MAP = {
    0: "A",
    1: "E"
}

print("Loading CNN language classifier...")

model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)

model.load_state_dict(torch.load(
    r"C:\Users\Office\PycharmProjects\deadMan\cnn language classifier files\model\language_classifier.pth",
    map_location=DEVICE
))

model = model.to(DEVICE)
model.eval()

print("CNN classifier loaded.")

# Image transform for classifier
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# =========================
# CLASSIFY LANGUAGE
# =========================

def classify_crop(crop):

    if crop.size == 0:
        return "A"

    img = transform(crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    return CLASS_MAP[pred]

# =========================
# DETECT FUNCTION
# =========================

def detect(image, image_name, output_dir):

    h, w = image.shape[:2]
    results = ocr.ocr(image)

    rows = []

    if results[0] is None:
        return rows

    for line in results[0]:

        box = line[0]
        full_text = line[1][0]
        conf = line[1][1]

        if conf < 0.6:
            continue

        xmin = int(box[0][0])
        ymin = int(box[0][1])
        xmax = int(box[2][0])
        ymax = int(box[2][1])

        line_crop = image[ymin:ymax, xmin:xmax]

        # Split into words
        words = full_text.strip().split()

        if len(words) == 0:
            continue

        total_chars = sum(len(w) for w in words)

        if total_chars == 0:
            continue

        current_x = xmin

        for word in words:

            proportion = len(word) / total_chars
            word_width = int((xmax - xmin) * proportion)

            word_xmin = current_x
            word_xmax = current_x + word_width

            word_crop = image[ymin:ymax, word_xmin:word_xmax]

            lang = classify_crop(word_crop)

            if lang == "A":
                current_x += word_width
                continue

            # Keep English word
            cv2.rectangle(image,
                          (word_xmin, ymin),
                          (word_xmax, ymax),
                          (0, 255, 0), 2)

            cv2.putText(image, word,
                        (word_xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0), 2)

            rows.append([
                image_name, w, h,
                "OCR", word,
                word_xmin, ymin,
                word_xmax, ymax,
                conf
            ])

            current_x += word_width

    save_path = os.path.join(output_dir, f"annotated_{image_name}")
    cv2.imwrite(save_path, image)

    return rows

# =========================
# PROCESS FOLDER
# =========================

def process_folder(folder, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "ocr_results_1.csv")

    all_rows = []

    for file in os.listdir(folder):

        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        print("Processing:", file)

        path = os.path.join(folder, file)
        image = cv2.imread(path)

        if image is None:
            continue

        rows = detect(image, file, output_dir)
        all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows, columns=[
            "image", "width", "height", "type", "text",
            "xmin", "ymin", "xmax", "ymax", "confidence"
        ])
        df.to_csv(csv_path, index=False)
        print("CSV saved:", csv_path)
    else:
        print("No valid English text detected.")

    print("DONE")

# =========================
# RUN
# =========================

if __name__ == "__main__":

    input_folder = r"C:\Users\Office\PycharmProjects\deadMan\input images"
    output_folder = r"C:\Users\Office\PycharmProjects\deadMan\text extraction using paddle OCR\cnn classifier to kB version\version 2 with segmentation\output results"

    process_folder(input_folder, output_folder)