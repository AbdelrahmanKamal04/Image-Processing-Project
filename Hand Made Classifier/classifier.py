# import easyocr
# import skimage.io as io


# reader = easyocr.Reader(['ar'])  # Load Arabic model (downloads ~100MB on first use)
# result = reader.readtext('Dataset/alef.png')


# print("Detected:", result[0][1] if result else "No text found")
import easyocr

reader = easyocr.Reader(['ar'])

results = reader.readtext(
    'Dataset/geem.png',
    min_size=10,           # detect very small text
    text_threshold=0.1,    # lower confidence threshold
    low_text=0.1,          # detect faint/low-contrast text
    link_threshold=0.1,    # for fragmented glyphs
    width_ths=1.0,         # treat wide spacing as one line (not needed here, but safe)
    mag_ratio=2            # upscale image for better detail
)

for (bbox, text, prob) in results:
    print(f"Detected: '{text}' | Confidence: {prob:.2f}")

# 