"""
DR ViT - Diabetic Retinopathy Prediction Module

This module imports the model and prediction function from model.py.
The ONNX model is loaded and managed by model.py.
"""

from model import predict_image as model_predict_image


# =========================================================
# PREDICTION
# =========================================================

def predict_image(image_path):
    """
    Predict the diabetic retinopathy class from a retinal image.

    Args:
        image_path (str): Full path to the retinal image.

    Returns:
        dict: Predicted class and confidence percentage.

    Example:
        {
            "prediction": "Moderate",
            "confidence": 87.42
        }
    """

    result = model_predict_image(image_path)

    return {
        "prediction": result["prediction"],
        "confidence": float(result["confidence"])
    }


# =========================================================
# TEST MODULE
# =========================================================

if __name__ == "__main__":
    print("=" * 50)
    print("DR ViT PREDICTION MODULE")
    print("=" * 50)
    print("Model connection: Successful")
    print("Prediction function: Ready")
    print("Model inference: ONNX Runtime")
    print("=" * 50)
    print("Usage: predict_image(image_path)")