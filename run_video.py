import cv2
import numpy as np
from rembg import remove, new_session
import os

input_video = "sample_video.mp4"
output_video = "output_velocity_blur.mp4"

if not os.path.exists(input_video):
    print("❌ एरर: मुझे 'sample_video.mp4' नहीं मिली!")
    exit()

print("🔥 SMX AI Video Studio: 3-सेकंड फ़ास्ट टेस्ट मोड चालू...")
session = new_session()

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps == 0:
    fps = 30

# सिर्फ 75 फ्रेम्स (लगभग 2.5 से 3 सेकंड) टेस्ट करेंगे
total_test_frames = 75

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print(f"🎬 कुल टेस्ट फ्रेम्स: {total_test_frames} | रेंडरिंग जारी है...")

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= total_test_frames:
        break

    # 1. AI बॉडी मास्क
    small_frame = cv2.resize(frame, (480, int(height * (480 / width))))
    mask_small = remove(small_frame, session=session, only_mask=True)
    mask = cv2.resize(mask_small, (width, height))

    alpha = mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
    alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

    # 2. बैकग्राउंड ब्लर + सिनेमैटिक कलर्स
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
    vibrant_bg = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    blurred_bg = cv2.GaussianBlur(vibrant_bg, (55, 55), 0)

    # 3. शार्प बॉडी + ब्लर बैकग्राउंड
    processed_frame = (frame * alpha_3d + blurred_bg * (1.0 - alpha_3d)).astype(np.uint8)

    # 4. वेलोसिटी इफ़ेक्ट
    cycle = (frame_idx % int(fps * 2)) / (fps * 2)
    if cycle < 0.3:
        if frame_idx % 2 == 0:
            out.write(processed_frame)
    elif 0.3 <= cycle < 0.7:
        for _ in range(3):
            out.write(processed_frame)
    else:
        out.write(processed_frame)

    frame_idx += 1
    percent = int((frame_idx / total_test_frames) * 100)
    print(f"⏳ प्रगति: {percent}% पूरी हुई ({frame_idx}/{total_test_frames})")

cap.release()
out.release()
print(f"✅ सफलता! टेस्ट वीडियो तैयार हो गई -> {output_video}")