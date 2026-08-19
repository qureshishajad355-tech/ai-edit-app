import cv2
import numpy as np
from rembg import remove
import os

print("🔥 SMX AI Studio: इंजन चालू हो रहा है...")

def process_image():
    # पक्का कर लें कि आपकी फोटो का नाम यही है
    input_file = "sample.jpg" 
    output_file = "output_result.jpg"
    
    if not os.path.exists(input_file):
        print(f"❌ एरर: मुझे '{input_file}' नाम की फोटो नहीं मिली!")
        print("💡 टिप: अपनी फोटो का नाम बदल कर 'sample.jpg' कर दो।")
        return

    print("📸 फोटो लोड हो रही है...")
    img = cv2.imread(input_file)
    
    print("⏳ AI इंसान को बैकग्राउंड से अलग कर रहा है (रुकें)...")
    nobg = remove(img)
    alpha = nobg[:, :, 3] / 255.0
    alpha = cv2.GaussianBlur(alpha, (9, 9), 0)
    mask_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

    # स्किन स्मूथिंग
    person = cv2.bilateralFilter(img, d=7, sigmaColor=60, sigmaSpace=60)
    
    # बैकग्राउंड ब्लर + कलर बूस्ट
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
    vibrant_bg = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    blurred_bg = cv2.GaussianBlur(vibrant_bg, (99, 99), 0)

    # कंबाइन
    output = (person * mask_3d + blurred_bg * (1.0 - mask_3d)).astype(np.uint8)
    
    cv2.imwrite(output_file, output)
    print(f"✅ सफलता! DSLR फोटो तैयार हो गई -> {output_file}")

if __name__ == "__main__":
    process_image()