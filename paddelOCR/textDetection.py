
import cv2

import os

import pandas as pd

import re

from paddleocr import PaddleOCR

#  LOAD MODEL

print("Loading PaddleOCR model...")

print("Loading PaddleOCR...")

ocr = PaddleOCR(lang='en')

print("Model loaded successfully.")


#  FILTER

def valid_english(text, conf):

    text = text.strip()

    if conf < 0.6:

        return False

    # remove arabic unicode

    if re.search(r'[\u0600-\u06FF]', text):

        return False

    # must contain english letters

    if not re.search(r'[A-Za-z]', text):

        return False

    # remove garbage

    if len(text) <= 2:

        return False

    return True

#  DETECT FUNCTION

def detect(image, image_name, output_dir):

    h, w = image.shape[:2]

    results = ocr.ocr(image)

    rows = []

    if results[0] is None:

        return rows

    for line in results[0]:

        box = line[0]

        text = line[1][0]

        conf = line[1][1]

        if not valid_english(text, conf):

            continue

        xmin = int(box[0][0])

        ymin = int(box[0][1])

        xmax = int(box[2][0])

        ymax = int(box[2][1])

        # draw bbox

        cv2.rectangle(image, (xmin ,ymin), (xmax ,ymax), (0 ,255 ,0), 2)

        cv2.putText(image, text, (xmin ,ymin -5),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0 ,255 ,0), 2)

        rows.append([

            image_name, w, h, "OCR", text,

            xmin, ymin, xmax, ymax, conf

        ])

    # save annotated image

    save_path = os.path.join(output_dir, f"annotated_{image_name}")

    cv2.imwrite(save_path, image)

    return rows

#  PROCESS FOLDER

def process_folder(folder, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "ocr_results_1.csv")

    all_rows = []

    for file in os.listdir(folder):

        if not file.lower().endswith((".jpg" ,".png" ,".jpeg")):

            continue

        print("Processing:", file)

        path = os.path.join(folder, file)

        image = cv2.imread(path)

        if image is None:

            continue

        rows = detect(image, file, output_dir)

        all_rows.extend(rows)

    # save csv

    if all_rows:

        df = pd.DataFrame(all_rows, columns=[

            "image" ,"width" ,"height" ,"type" ,"text",

            "xmin" ,"ymin" ,"xmax" ,"ymax" ,"confidence"

        ])

        df.to_csv(csv_path, index=False)

        print("CSV saved:", csv_path)

    else:

        print("No valid text detected in any images.")

    print("DONE")

#  RUN

if __name__ == "__main__":

    input_folder = r"C:\Users\Office\PycharmProjects\local_ocr_paddle_project\paddelOCR\inputImages"

    output_folder = r"C:\Users\Office\PycharmProjects\local_ocr_paddle_project\paddelOCR\outputResults"

    process_folder(input_folder, output_folder)
