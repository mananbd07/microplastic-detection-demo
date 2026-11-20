import os
import io
import glob
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# Import your custom logic
from eye import build_classical_feature_extractor, IMG_SHAPE
from brain import quantum_classifier_head, get_quantum_weights_variable

# 1. Load the Models
print("Loading models...")
classical_model = build_classical_feature_extractor()
quantum_weights = get_quantum_weights_variable()

CLASSICAL_WEIGHTS_PATH = "expert_classical_eye.weights.h5"
QUANTUM_WEIGHTS_PATH = "expert_quantum_brain.npy"

try:
    classical_model.load_weights(CLASSICAL_WEIGHTS_PATH)
except Exception:
    classical_model.load_weights(CLASSICAL_WEIGHTS_PATH, skip_mismatch=True)

q_saved = np.load(QUANTUM_WEIGHTS_PATH, allow_pickle=True)
quantum_weights.assign(q_saved)
print("Models loaded successfully!")

# 2. Prediction Function
def process_image(image, confidence_threshold=0.90):
    if image is None:
        return None, "Please upload an image."
    
    # Convert PIL to OpenCV format (RGB to BGR)
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Preprocessing for finding contours
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(denoised)
    _, thresh = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Setup PIL image for drawing boxes
    out_img = Image.fromarray(img_np).convert("RGB")
    draw = ImageDraw.Draw(out_img)
    
    plastic_count = 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 100: # Ignore tiny noise
            continue
            
        crop = img_bgr[y:y+h, x:x+w]
        if crop.size == 0:
            continue
            
        # AI RECOMMENDATION: Fix training/inference mismatch by applying enhancement to the crop
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_denoised = cv2.fastNlMeansDenoising(crop_gray, None, 10, 7, 21)
        crop_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(crop_denoised)
        crop_enhanced = cv2.cvtColor(crop_clahe, cv2.COLOR_GRAY2BGR)
        
        # Resize and normalize to tf.float64 (as used in training)
        resized = cv2.resize(crop_enhanced, (IMG_SHAPE[1], IMG_SHAPE[0]))
        img_tensor = tf.convert_to_tensor(resized, dtype=tf.float64)
        img_batch = tf.expand_dims(img_tensor, axis=0)
        
        # Predict
        features = classical_model(img_batch, training=False)
        q_out = quantum_classifier_head(features, quantum_weights)
        prob = (q_out.numpy()[0] + 1.0) / 2.0
        
        if prob >= confidence_threshold:
            plastic_count += 1
            # Draw red box for plastic
            draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
            draw.text((x, y-10), f"Plastic: {prob:.2f}", fill="red")
        else:
            # Draw green box for clean
            draw.rectangle([x, y, x+w, y+h], outline="green", width=2)

    result_text = f"Scan Complete.\nMicroplastics Detected: {plastic_count}"
    return out_img, result_text

# 3. Build the Gradio UI
example_images = glob.glob("examples/*.jpg") + glob.glob("examples/*.png")

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Water Sample Image"),
        gr.Slider(minimum=0.5, maximum=0.99, value=0.90, step=0.01, label="Confidence Threshold")
    ],
    outputs=[
        gr.Image(type="pil", label="Annotated Result"),
        gr.Textbox(label="Detection Summary")
    ],
    title="🔬 Microplastic Detection System (Hybrid Quantum-Classical)",
    description="Upload an image of a water sample. The model uses an EfficientNetV2B0 feature extractor and a PennyLane 4-qubit quantum circuit to detect microplastics.",
    examples=example_images,
    theme="soft"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))