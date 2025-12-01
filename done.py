#!/usr/bin/env python3
"""
classic_plate_reader.py

Classical image-processing license-plate reader (pure software, no neural nets).
Input: an image file (png/jpg). Output: recognized plate(s) and decision OPEN/KEEP CLOSED.

Dependencies:
    pip install opencv-python numpy pillow

Author: Generated scaffold (classical IP pipeline)
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import os
import sys
import argparse
import logging
from typing import List, Tuple, Optional, Dict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("ClassicPlateReader")

# -----------------------------
# Configuration / assumptions
# -----------------------------
CONFIG = {
    "min_plate_area": 1500,    # minimal area (pixels) of bounding rect to consider
    "max_plate_area": 50000,   # maximal area
    "min_plate_aspect": 2.0,   # width / height
    "max_plate_aspect": 6.5,
    "debug_save_dir": "debug_outputs",
    "template_font": None,     # if None, PIL will use default font
    "char_set": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "plate_length_min": 5,
    "plate_length_max": 10,
    "visualize": False,
    "use_clahe": True,
    "save_intermediates": True,
    "template_font_sizes": [36, 42, 48, 56],  # font sizes to generate templates for matching
    "template_scale_factors": [0.8, 1.0, 1.2],
    "debug_verbose": False,
}

# Ensure debug dir
os.makedirs(CONFIG["debug_save_dir"], exist_ok=True)


# -----------------------------
# Utilities
# -----------------------------
def save_debug_image(img, name_prefix):
    if not CONFIG["save_intermediates"]:
        return
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    filename = f"{name_prefix}_{timestamp}.png"
    path = os.path.join(CONFIG["debug_save_dir"], filename)
    cv2.imwrite(path, img)
    if CONFIG["debug_verbose"]:
        log.info("Saved debug image: %s", path)


def show_im(title, img, scale=1.0):
    if not CONFIG["visualize"]:
        return
    h, w = img.shape[:2]
    disp = cv2.resize(img, (int(w*scale), int(h*scale)))
    cv2.imshow(title, disp)
    cv2.waitKey(0)


def normalize_plate_string(s: str) -> str:
    """Normalize recognized plate: uppercase, remove non-alnum."""
    return "".join(ch for ch in s.upper() if ch.isalnum())


# -----------------------------
# Plate candidate detection
# -----------------------------
class PlateDetector:
    def __init__(self):
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Return enhanced grayscale image for detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # apply CLAHE to boost contrast if requested
        if CONFIG["use_clahe"]:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        # bilateral filter to preserve edges
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        return gray

    def detect_candidates(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Returns list of (warped_plate_image, bbox (x,y,w,h) in original image, score)
        Score is a heuristic indicating how plate-like the region is.
        """
        gray = self.preprocess(image)
        # Edge detection
        edged = cv2.Canny(gray, 60, 200)
        save_debug_image(edged, "edges")

        # Morphological close to connect plate edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        save_debug_image(closed, "closed")

        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = gray.shape
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < CONFIG["min_plate_area"] or area > CONFIG["max_plate_area"]:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.06 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            if h == 0:  # guard
                continue
            aspect = w / float(h)
            if aspect < CONFIG["min_plate_aspect"] or aspect > CONFIG["max_plate_aspect"]:
                continue

            # Get rotated rect and warp perspective to get straight plate
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            warped = self.four_point_transform(image, box)
            # compute internal heuristics: edge density, contrast, stroke density
            score = self.candidate_score(warped)
            candidates.append((warped, (x, y, w, h), score))

        # Sort by score descending
        candidates.sort(key=lambda t: t[2], reverse=True)
        return candidates

    @staticmethod
    def candidate_score(warped: np.ndarray) -> float:
        """Compute a heuristic score of how plate-like the warped image is."""
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h == 0 or w == 0:
            return 0.0
        # Edge density inside region
        edges = cv2.Canny(gray, 50, 200)
        edge_density = np.sum(edges > 0) / float(w * h)
        # Contrast via stddev
        contrast = float(np.std(gray)) / 255.0
        # vertical stroke density: compute vertical sobel and its mean magnitude
        sobelx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        stroke_density = np.mean(sobelx) / 255.0
        # final weighted score (weights tuned heuristically)
        score = (edge_density * 0.5) + (contrast * 0.3) + (stroke_density * 0.2)
        # multiply by aspect closeness to ideal (approx 4.0 typical)
        aspect = w / float(h) if h > 0 else 1.0
        aspect_score = 1.0 - abs(aspect - 4.0) / 4.0  # in range roughly [-..,1]
        score *= max(0.0, aspect_score)
        return score

    @staticmethod
    def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Perform perspective transform on quadrilateral pts (4x2)."""
        # order points: tl,tr,br,bl
        rect = PlateDetector.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    @staticmethod
    def order_points(pts: np.ndarray):
        # pts shape (4,2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


# -----------------------------
# Plate preprocessing & segmentation
# -----------------------------
class PlateProcessor:
    def __init__(self):
        pass

    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """Take warped plate color image and return a binarized image suitable for segmentation."""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        # Resize plate to standard height for consistent segmentation
        target_h = 120
        scale = target_h / float(gray.shape[0])
        new_w = int(gray.shape[1] * scale)
        gray = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
        # Contrast & adaptive threshold
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # morphological top-hat to emphasize lighter characters on dark background or vice versa
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        # Use adaptive threshold on tophat or gray
        th = cv2.adaptiveThreshold(tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 12)
        # Small opening to remove speckles
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel2)
        save_debug_image(opening, "plate_binarized")
        return opening

    def segment_characters(self, bin_img: np.ndarray) -> List[np.ndarray]:
        """Segment characters using connected components + vertical projection heuristics.
        Returns list of character images (each as binary image), left-to-right order.
        """
        # find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
        h, w = bin_img.shape
        candidates = []
        for i in range(1, num_labels):  # skip background
            x, y, ww, hh, area = stats[i]
            # basic filtering heuristics for char shapes
            if area < 60 or hh < 20 or ww < 8:
                continue
            aspect = ww / float(hh)
            if aspect > 1.2:  # characters are usually taller than wide; wide blobs might be joined characters
                # But allow some flexibility: if large area might be two chars, consider splitting later
                pass
            candidates.append((x, y, ww, hh, area))
        # If no candidates found (maybe characters are touching): fallback to vertical projection
        if not candidates:
            return self._segment_by_projection(bin_img)
        # Otherwise, sort left-to-right and extract images
        candidates = sorted(candidates, key=lambda r: r[0])
        char_imgs = []
        for (x, y, ww, hh, _) in candidates:
            roi = bin_img[y:y+hh, x:x+ww]
            # pad to square-ish
            pad = max(int(0.2*hh), 2)
            roi_padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            # resize to standard box
            roi_norm = cv2.resize(roi_padded, (40, 60), interpolation=cv2.INTER_AREA)
            char_imgs.append(roi_norm)
        return char_imgs

    def _segment_by_projection(self, bin_img):
        """Fallback segmentation: vertical projection peaks to find char columns."""
        h, w = bin_img.shape
        vertical = np.sum(bin_img > 0, axis=0)
        # smooth vertical histogram
        kernel = np.ones(5)/5.0
        v_smooth = np.convolve(vertical, kernel, mode='same')
        thresh = max(3, int(0.05 * h))  # minimal stroke count to consider column active
        # detect runs of columns above threshold
        in_run = False
        runs = []
        run_start = 0
        for i, v in enumerate(v_smooth):
            if v > thresh and not in_run:
                in_run = True
                run_start = i
            elif v <= thresh and in_run:
                in_run = False
                runs.append((run_start, i-1))
        if in_run:
            runs.append((run_start, w-1))
        # filter narrow runs and create boxes
        boxes = []
        for s, e in runs:
            ww = e - s + 1
            if ww < 4:
                continue
            # extract ROI vertically across full height
            roi = bin_img[:, s:e+1]
            # find tight vertical crop of content
            rows = np.where(np.sum(roi, axis=1) > 0)[0]
            if len(rows) == 0:
                continue
            y0, y1 = rows[0], rows[-1]
            roi_crop = roi[max(0, y0-2):min(h, y1+3), :]
            pad = 4
            roi_pad = cv2.copyMakeBorder(roi_crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            roi_norm = cv2.resize(roi_pad, (40, 60), interpolation=cv2.INTER_AREA)
            boxes.append(roi_norm)
        return boxes


# -----------------------------
# Template generator & matcher (classical OCR)
# -----------------------------
class TemplateOCR:
    def __init__(self, char_set: str):
        self.char_set = char_set
        self.templates = {}  # char -> list of template images (numpy arrays)
        self.generate_templates()

    def generate_templates(self):
        """Programmatically generate templates for all characters using PIL fonts."""
        # Try to use a system font; if not available, PIL default will be used (monospace)
        for ch in self.char_set:
            self.templates[ch] = []
            for size in CONFIG["template_font_sizes"]:
                for scale in CONFIG["template_scale_factors"]:
                    img = self._render_char(ch, int(size * scale))
                    arr = np.array(img)
                    # binary: white on black -> invert later if needed
                    _, bin_img = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY)
                    # normalize to same size as segments
                    norm = cv2.resize(bin_img, (40, 60), interpolation=cv2.INTER_AREA)
                    self.templates[ch].append(norm)

    def _render_char(self, ch: str, font_size: int) -> Image.Image:
        # make image with transparent background and render character in black on white
        img = Image.new('L', (80, 120), color=255)
        draw = ImageDraw.Draw(img)
        # load default font
        try:
            # attempt to get an available truetype font
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
        w, h = draw.textsize(ch, font=font)
        x = (img.width - w) // 2
        y = (img.height - h) // 2
        draw.text((x, y), ch, font=font, fill=0)
        return img

    def recognize(self, char_img: np.ndarray) -> Tuple[str, float]:
        """
        Recognize a single character image via normalized cross-correlation to templates.
        Returns best_char, confidence (0..100)
        """
        # Ensure char_img is same size (40x60)
        img = char_img
        if img.shape != (60, 40):
            img = cv2.resize(img, (40, 60), interpolation=cv2.INTER_AREA)

        # Binarize & ensure white glyph on black background (we use white-on-black)
        # Our segmenters produce binary where foreground=255; we desire templates same style.
        if np.mean(img) > 127:
            # churn to get binary consistent
            _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        else:
            _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        best_char = '?'
        best_score = -1.0
        # iterate templates for each char
        for ch, templ_list in self.templates.items():
            for templ in templ_list:
                # both are 40x60 binary images
                # compute normalized match score: correlation
                # convert to float and normalize
                a = img_bin.astype(np.float32) / 255.0
                b = templ.astype(np.float32) / 255.0
                # use zero-mean normalized cross-correlation
                a_mean = a.mean()
                b_mean = b.mean()
                num = ((a - a_mean) * (b - b_mean)).sum()
                den = math.sqrt(((a - a_mean) ** 2).sum() * ((b - b_mean) ** 2).sum() + 1e-8)
                corr = float(num / (den + 1e-9))
                # corr in [-1,1]; map to 0..100
                score = (corr + 1.0) * 50.0
                if score > best_score:
                    best_score = score
                    best_char = ch
        return best_char, float(best_score)

    def recognize_string(self, char_imgs: List[np.ndarray]) -> Tuple[str, float, List[float]]:
        """Recognize list of character images, return string, avg confidence and per-char confidences."""
        chars = []
        confidences = []
        for ci in char_imgs:
            ch, conf = self.recognize(ci)
            chars.append(ch)
            confidences.append(conf)
        if not chars:
            return "", 0.0, []
        avg_conf = sum(confidences) / len(confidences)
        return "".join(chars), avg_conf, confidences


# -----------------------------
# Postprocessing heuristics
# -----------------------------
class PlatePostProcessor:
    def __init__(self):
        # ambiguous substitutions to try for corrections
        self.substitutions = {
            '0': ['O'],
            'O': ['0'],
            '1': ['I', 'L'],
            'I': ['1', 'L'],
            '5': ['S'],
            'S': ['5'],
            '2': ['Z'],
            'Z': ['2']
        }

    def apply_rules(self, s: str) -> Tuple[str, float]:
        """Apply simple heuristics to validate and slightly correct a plate string.
        Returns corrected string and a penalty-adjusted confidence (0..100).
        """
        s = normalize_plate_string(s)
        if not s:
            return "", 0.0
        # clamp length
        if len(s) < CONFIG["plate_length_min"] or len(s) > CONFIG["plate_length_max"]:
            # penalize
            penalty = 0.6
        else:
            penalty = 1.0
        # basic character sanity: ensure only allowed characters
        allowed = set(CONFIG["char_set"])
        cleaned = []
        char_penalties = []
        for ch in s:
            if ch in allowed:
                cleaned.append(ch)
                char_penalties.append(1.0)
            else:
                # try substitution heuristics (single char confusion)
                replaced = False
                for sub in self.substitutions.get(ch, []):
                    if sub in allowed:
                        cleaned.append(sub)
                        char_penalties.append(0.8)
                        replaced = True
                        break
                if not replaced:
                    # drop char with big penalty
                    char_penalties.append(0.2)
                    # keep as '?'
                    cleaned.append('?')
        final = "".join(cleaned)
        avg_char_pen = sum(char_penalties) / len(char_penalties) if char_penalties else 0.0
        final_conf_factor = penalty * avg_char_pen
        return final, final_conf_factor * 100.0


# -----------------------------
# Gate Controller (mock)
# -----------------------------
class GateControllerMock:
    def __init__(self):
        self.state = "closed"

    def open_gate(self):
        if self.state != "open":
            log.info("GATE ACTION: OPEN GATE")
            self.state = "open"

    def close_gate(self):
        if self.state != "closed":
            log.info("GATE ACTION: CLOSE GATE")
            self.state = "closed"


# -----------------------------
# Main system orchestrator
# -----------------------------
class ClassicPlateReader:
    def __init__(self, authorized_db: List[str]):
        """
        authorized_db: list of authorized plate strings (normalized)
        """
        self.detector = PlateDetector()
        self.plate_processor = PlateProcessor()
        self.ocr = TemplateOCR(CONFIG["char_set"])
        self.postproc = PlatePostProcessor()
        self.authorized = set(normalize_plate_string(p) for p in authorized_db)
        self.gate = GateControllerMock()

    def process_image(self, image_path: str) -> Dict:
        log.info("Processing image: %s", image_path)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Cannot read image: " + image_path)
        orig = image.copy()
        # detect plate candidates
        candidates = self.detector.detect_candidates(image)
        results = []
        if not candidates:
            log.info("No plate candidates found.")
        for idx, (plate_img, bbox, score) in enumerate(candidates):
            log.info("Candidate #%d score=%.3f bbox=%s", idx+1, score, bbox)
            save_debug_image(plate_img, f"candidate_{idx+1}")
            bin_plate = self.plate_processor.preprocess_plate(plate_img)
            char_imgs = self.plate_processor.segment_characters(bin_plate)
            log.info("Segmented %d character candidate(s).", len(char_imgs))
            save_debug_image(bin_plate, f"candidate_{idx+1}_bin")
            # recognize
            recog_str, avg_conf, char_confs = self.ocr.recognize_string(char_imgs)
            log.info("Raw recognition: %s (avg_conf=%.1f)", recog_str, avg_conf)
            # postprocess
            post_str, conf_factor = self.postproc.apply_rules(recog_str)
            # final confidence = avg_conf * conf_factor / 100
            final_conf = avg_conf * (conf_factor / 100.0) if avg_conf > 0 else conf_factor
            final_conf = min(100.0, final_conf)
            normalized = normalize_plate_string(post_str)
            # authorization check
            authorized = normalized in self.authorized and len(normalized) > 0
            results.append({
                "candidate_index": idx+1,
                "bbox": bbox,
                "initial_score": score,
                "raw_recognition": recog_str,
                "per_char_confidences": char_confs,
                "post_recognition": post_str,
                "final_confidence": final_conf,
                "authorized": authorized
            })
            log.info("Post-processed: %s final_confidence=%.2f authorized=%s",
                     post_str, final_conf, authorized)
            # If authorized and confidence above threshold -> open gate
            if authorized and final_conf > 30.0:
                self.gate.open_gate()
            else:
                log.info("Candidate not authorized or low confidence.")
        # If no candidate triggered open, ensure gate closed
        if not any(r["authorized"] and r["final_confidence"] > 30.0 for r in results):
            self.gate.close_gate()
        # print details to stdout per your request
        self.print_report(image_path, results)
        return {"image": image_path, "results": results}

    @staticmethod
    def print_report(image_path: str, results: List[Dict]):
        print("\n=== Gate Access Report ===")
        print(f"Image: {image_path}")
        print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
        if not results:
            print("No plate detected.")
        else:
            for r in results:
                print("\n--- Candidate #{candidate_index} ---".replace("{candidate_index}", str(r["candidate_index"])))
                print(f"Bounding box (x,y,w,h): {r['bbox']}")
                print(f"Initial plate-likeness score: {r['initial_score']:.3f}")
                print(f"Raw OCR: {r['raw_recognition']}")
                print(f"Per-char confidences: {['{:.1f}'.format(c) for c in r['per_char_confidences']]}")
                print(f"Post-processed recognition: {r['post_recognition']}")
                print(f"Final confidence: {r['final_confidence']:.2f}/100")
                print(f"Authorized: {r['authorized']}")
                print("------------------------------")
        print("\nDecision: " + ("OPEN GATE" if any(r['authorized'] and r['final_confidence'] > 30.0 for r in results) else "KEEP GATE CLOSED"))
        print("===========================\n")


# -----------------------------
# Command line interface
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Classical License Plate Reader (no NN).")
    p.add_argument("--image", required=True, help="Input image path (PNG/JPG)")
    p.add_argument("--authorized", nargs="*", default=["ABC123", "XYZ999", "TEST001"], help="List of authorized plates (space separated)")
    p.add_argument("--visualize", action='store_true', help="Show intermediate images (debug UI)")
    p.add_argument("--save-debug", action='store_true', help="Save debug intermediate images to debug_outputs/")
    p.add_argument("--debug-verbose", action='store_true', help="Verbose debug prints")
    return p.parse_args()


def main():
    args = parse_args()
    CONFIG["visualize"] = args.visualize
    CONFIG["save_intermediates"] = args.save_debug
    CONFIG["debug_verbose"] = args.debug_verbose
    # Normalize provided authorized list
    auth_list = [normalize_plate_string(p) for p in args.authorized]

    reader = ClassicPlateReader(auth_list)
    reader.process_image(args.image)

if __name__ == "__main__":
    main()
