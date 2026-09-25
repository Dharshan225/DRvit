import os
import numpy as np
import onnxruntime as ort

from torchvision import transforms
from PIL import Image


# =========================================================
# MODEL CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "DR_ViT_model.onnx"
)


# =========================================================
# CLASS NAMES
# =========================================================

# EXACT SAME ORDER AS ORIGINAL model.py

CLASS_NAMES = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]


# =========================================================
# LOAD ONNX MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"ONNX model not found at: {MODEL_PATH}"
    )


session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)


# =========================================================
# INPUT / OUTPUT INFORMATION
# =========================================================

INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name


print("========================================")
print("ONNX MODEL LOADED SUCCESSFULLY")
print("========================================")
print("Model:", MODEL_PATH)
print("Input name:", INPUT_NAME)
print("Output name:", OUTPUT_NAME)
print("Input shape:", session.get_inputs()[0].shape)
print("Output shape:", session.get_outputs()[0].shape)
print("========================================")


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

])


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image_path):

    # -----------------------------------------------------
    # Open image
    # -----------------------------------------------------

    image = Image.open(image_path).convert("RGB")


    # -----------------------------------------------------
    # Preprocess
    # -----------------------------------------------------

    image = transform(image)

    # Add batch dimension
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


    # Get output for first image
    outputs = outputs[0][0]


    # -----------------------------------------------------
    # SOFTMAX
    # -----------------------------------------------------

    probabilities = np.exp(
        outputs - np.max(outputs)
    )

    probabilities = probabilities / np.sum(
        probabilities
    )


    # -----------------------------------------------------
    # FIND PREDICTED CLASS
    # -----------------------------------------------------

    predicted_index = int(
        np.argmax(probabilities)
    )


    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    confidence = float(
        probabilities[predicted_index]
    )


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    return {

        "prediction": CLASS_NAMES[predicted_index],

        "confidence": round(
            confidence * 100,
            2
        )

    }