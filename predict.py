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
# Same class order as your original model.py

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


# =========================================================
# LOAD ONNX MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"ONNX model not found: {MODEL_PATH}"
    )


session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)


# Get ONNX input/output names

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
# PREDICTION
# =========================================================

def predict_image(image_path):

    # -----------------------------------------------------
    # Open image
    # -----------------------------------------------------

    image = Image.open(image_path).convert("RGB")


    # -----------------------------------------------------
    # Preprocess image
    # -----------------------------------------------------

    image = transform(image)


    # Add batch dimension

    image = image.unsqueeze(0)


    # Convert PyTorch tensor to NumPy

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


    # Get model output

    outputs = outputs[0]


    # -----------------------------------------------------
    # SOFTMAX
    # -----------------------------------------------------

    probabilities = np.exp(
        outputs - np.max(outputs, axis=1, keepdims=True)
    )

    probabilities = probabilities / np.sum(
        probabilities,
        axis=1,
        keepdims=True
    )


    # -----------------------------------------------------
    # GET PREDICTION
    # -----------------------------------------------------

    predicted = np.argmax(
        probabilities,
        axis=1
    )[0]


    confidence = probabilities[
        0,
        predicted
    ]


    # -----------------------------------------------------
    # RETURN SAME FORMAT AS OLD predict.py
    # -----------------------------------------------------

    return {

        "prediction": CLASS_NAMES[predicted],

        "confidence": round(
            float(confidence) * 100,
            2
        )

    }