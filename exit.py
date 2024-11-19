import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import pytesseract
from datetime import datetime

pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO('best.pt')

area = [(35, 375), (16, 456), (1015, 451), (965, 378)]

cap = cv2.VideoCapture('mycarplate.mp4')

# Load existing data
output_file = "car_plate_data.xlsx"
try:
    df = pd.read_excel(output_file)
except FileNotFoundError:
    print("No data file found. Please run the entry script first.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    boxes = results[0].boxes.data
    px = pd.DataFrame(boxes).astype("float")

    for _, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)

        if result >= 0:
            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 15, 17, 17)

            text = pytesseract.image_to_string(gray).strip()
            text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '')

            if text in df["Car Number Plate ID"].values:
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df.loc[df["Car Number Plate ID"] == text, "Exit Time"] = current_datetime

    if cv2.waitKey(1) & 0xFF == 27:
        break

df.to_excel(output_file, index=False)
print(f"Exit data updated in {output_file}")

cap.release()
cv2.destroyAllWindows()
