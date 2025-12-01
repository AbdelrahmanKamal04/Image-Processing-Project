import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import re

# Buses & government vehicles (Not Completely accurate: Extra Numbers)

# Diplomatic vehicles (Not Completely accurate: Second Half Of Numbers Do Not Appear)

# Police vehicles (Not Completely accurate: Color Not Correct)
# Limousines & tourist buses (Not Completely accurate: Color Not Correct)
# Trucks (Not Completely accurate: Color Not Correct)
# Vehicles with unpaid customs (Not Completely accurate: Color Not Correct)

# Private vehicles & motorcycles (Correct Output)
# Taxis (Correct Output)

# === STEP 1: Load image ===
image_path = "Plate Types/car2.png"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("❌ Could not read image. Check your file path.")

# Resize for consistency
image = cv2.resize(image, (1000, int(image.shape[0] * 1000 / image.shape[1])))

# === STEP 2: Convert to HSV ===
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# === STEP 3: Define plate color ranges and vehicle types ===
plate_colors = {
    "Light Blue": {
        "type": "Private vehicles & motorcycles",
        "lower": np.array([90, 50, 70]),
        "upper": np.array([130, 255, 255])
    },
    "Orange": {
        "type": "Taxis",
        "lower": np.array([5, 100, 100]),
        "upper": np.array([25, 255, 255])
    },
    "Red": {
        "type": "Trucks",
        "lower": np.array([0, 100, 100]),
        "upper": np.array([10, 255, 255])
    },
    "Gray": {
        "type": "Buses & government vehicles",
        "lower": np.array([0, 0, 40]),
        "upper": np.array([180, 50, 180])
    },
    "Beige": {
        "type": "Limousines & tourist buses",
        "lower": np.array([10, 20, 150]),
        "upper": np.array([25, 70, 255])
    },
    "Green": {
        "type": "Diplomatic vehicles",
        "lower": np.array([40, 50, 70]),
        "upper": np.array([80, 255, 255])
    },
    "Yellow": {
        "type": "Vehicles with unpaid customs",
        "lower": np.array([25, 100, 100]),
        "upper": np.array([35, 255, 255])
    },
    "Dark Blue": {
        "type": "Police vehicles",
        "lower": np.array([110, 100, 50]),
        "upper": np.array([130, 255, 200])
    }
}

# === STEP 4: Create combined mask and detect dominant color ===
combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
color_match_counts = {}

for color_name, data in plate_colors.items():
    mask = cv2.inRange(hsv, data["lower"], data["upper"])
    count = cv2.countNonZero(mask)
    color_match_counts[color_name] = count
    combined_mask = cv2.bitwise_or(combined_mask, mask)

# Determine dominant color (max match)
dominant_color = max(color_match_counts, key=color_match_counts.get)
vehicle_type = plate_colors[dominant_color]["type"]

# === STEP 5: Morphological cleanup ===
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# === STEP 6: Find contours and pick the largest plausible rectangle ===
# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# plate_contour = None
# max_area = 0
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     aspect_ratio = w / float(h)
#     area = cv2.contourArea(cnt)
#     if 2 < aspect_ratio < 6 and 5000 < area < 200000:  # typical Egyptian plate proportions
#         if area > max_area:
#             max_area = area
#             plate_contour = cnt

# if plate_contour != None:
#     # === STEP 7: Deskew / perspective correction ===
#     rect = cv2.minAreaRect(plate_contour)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)

#     # Sort points for perspective transform
#     def order_points(pts):
#         rect = np.zeros((4, 2), dtype="float32")
#         s = pts.sum(axis=1)
#         rect[0] = pts[np.argmin(s)]  # top-left
#         rect[2] = pts[np.argmax(s)]  # bottom-right
#         diff = np.diff(pts, axis=1)
#         rect[1] = pts[np.argmin(diff)]  # top-right
#         rect[3] = pts[np.argmax(diff)]  # bottom-left
#         return rect

#     pts = order_points(box)
#     (tl, tr, br, bl) = pts

#     width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
#     height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

#     dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
#     M = cv2.getPerspectiveTransform(pts, dst)
#     plate = cv2.warpPerspective(image, M, (width, height))

#     # Draw bounding box on original image
#     cv2.drawContours(image, [box], 0, (0, 255, 0), 3)

# === STEP 8: OCR ===
gray_plate = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
reader = easyocr.Reader(['ar'])
results = reader.readtext(gray_plate)

# === STEP 9: Clean and print results ===
filtered_texts = []
for (bbox, text, conf) in results:
    if conf >= 0.3:
        clean = re.sub(r'[^\u0660-\u0669\u0600-\u06FF]', '', text)
        if clean and (clean != 'مصر') and (clean != 'الشرطة'):
            filtered_texts.append(clean)

plate_text = " ".join(filtered_texts)

print("\n✅ Final Detected Plate:", plate_text if plate_text else "[None detected]") 

print(f"✅ Detected Plate Color: {dominant_color}")

print(f"✅ Vehicle Type: {vehicle_type}\n")