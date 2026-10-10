import os
import numpy as np
import onnxruntime as ort

from torchvision import transforms
from PIL import Image


# =========================================================
# MODEL CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "DR_ViT_model.onnx"
)

MODEL_DATA_PATH = os.path.join(
    MODEL_DIR,
    "DR_ViT_model.onnx.data"
)


# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]


# =========================================================
# VERIFY MODEL FILES
# =========================================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"ONNX model not found: {MODEL_PATH}\n"
        "Run download_model.py to download the model."
    )

if not os.path.isfile(MODEL_DATA_PATH):
    raise FileNotFoundError(
        f"ONNX external weights not found: {MODEL_DATA_PATH}\n"
        "Ensure both ONNX files are downloaded."
    )


# =========================================================
# LOAD ONNX MODEL
# =========================================================

try:
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

except Exception as error:
    raise RuntimeError(
        f"Failed to load the ONNX model: {error}"
    ) from error


# =========================================================
# INPUT / OUTPUT INFORMATION
# =========================================================

INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name


print("=" * 50)
print("DR ViT ONNX MODEL LOADED SUCCESSFULLY")
print("=" * 50)
print(f"Model path: {MODEL_PATH}")
print(f"External weights: {MODEL_DATA_PATH}")
print(f"Input name: {INPUT_NAME}")
print(f"Output name: {OUTPUT_NAME}")
print(f"Input shape: {session.get_inputs()[0].shape}")
print(f"Output shape: {session.get_outputs()[0].shape}")
print("=" * 50)


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image_path):
    """
    Predict the diabetic retinopathy class for an image.

    Args:
        image_path (str): Path to the retinal image.

    Returns:
        dict: Predicted class and confidence percentage.
    """

    # -----------------------------------------------------
    # VERIFY IMAGE
    # -----------------------------------------------------

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # -----------------------------------------------------
    # OPEN IMAGE
    # -----------------------------------------------------

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    # -----------------------------------------------------
    # PREPROCESS IMAGE
    # -----------------------------------------------------

    image = transform(image)

    # Add batch dimension: [1, 3, 224, 224]
    image = image.unsqueeze(0)

    # Convert to NumPy float32
    image = image.numpy().astype(np.float32)

    # -----------------------------------------------------
    # ONNX INFERENCE
    # -----------------------------------------------------

    outputs = session.run(
        [OUTPUT_NAME],
        {
            INPUT_NAME: image
        }
    )

    # Extract class scores for the first image
    logits = outputs[0][0]

    # -----------------------------------------------------
    # SOFTMAX
    # -----------------------------------------------------

    shifted_logits = logits - np.max(logits)

    probabilities = np.exp(shifted_logits)

    probabilities = probabilities / np.sum(probabilities)

    # -----------------------------------------------------
    # PREDICTED CLASS
    # -----------------------------------------------------

    predicted_index = int(np.argmax(probabilities))

    prediction = CLASS_NAMES[predicted_index]

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    confidence = float(probabilities[predicted_index]) * 100

    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }