import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import pytesseract
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO('best.pt')

area = [(35, 375), (16, 456), (1015, 451), (965, 378)]

cap = cv2.VideoCapture('mycarplate.mp4')

processed_numbers = set()

# Load existing data or create a new DataFrame
output_file = "car_plate_data.xlsx"
try:
    df = pd.read_excel(output_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=["Car Number Plate ID", "Entry Time", "Exit Time"])

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
            
            if text not in processed_numbers:
                processed_numbers.add(text)
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df = pd.concat([df, pd.DataFrame([{"Car Number Plate ID": text, "Entry Time": current_datetime, "Exit Time": ""}])], ignore_index=True)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

df.to_excel(output_file, index=False)
print(f"Entry data saved to {output_file}")

cap.release()
cv2.destroyAllWindows()
