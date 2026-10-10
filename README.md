# DR ViT — Diabetic Retinopathy Detection

A Vision Transformer (ViT)-based deep learning application for classifying diabetic retinopathy from retinal fundus images.

## Overview

DR ViT is a medical image classification project that combines a Vision Transformer model with a Flask web application.

The application accepts retinal fundus images and predicts one of five diabetic retinopathy categories. The trained model is exported to ONNX format, and ONNX Runtime is used for inference.

The source code is hosted on GitHub, while the trained model files are hosted separately on Hugging Face.

## Features

- Retinal fundus image upload
- Five-class image classification
- Prediction confidence display
- Patient history interface
- Dashboard for viewing prediction records
- Flask-based web application
- ONNX Runtime inference
- SQLite database integration
- Automatic model download from Hugging Face

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Web Framework | Flask |
| Deep Learning Architecture | Vision Transformer (ViT) |
| Model Format | ONNX |
| Inference Engine | ONNX Runtime |
| Image Processing | Pillow, NumPy |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| Model Hosting | Hugging Face |

## Classification Categories

The application is designed to classify retinal images into five categories:

1. No DR
2. Mild
3. Moderate
4. Severe
5. Proliferative DR

**Note:** The category-to-index mapping must match the mapping used during model training and export.

## Model Information

- Architecture: Vision Transformer (ViT)
- Input resolution: 224 × 224 pixels
- Model format: ONNX
- Inference engine: ONNX Runtime
- Number of classes: 5
- Validation accuracy: Approximately 82.47%

The reported accuracy represents a validation result. It does not establish clinical diagnostic accuracy or suitability for medical decision-making.

## Project Structure

```text
DRvit/
├── app.py
├── model.py
├── predict.py
├── download_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── patient_history.html
├── static/
│   ├── style.css
│   ├── script.js
│   └── uploads/
└── models/
    ├── DR_ViT_model.onnx
    └── DR_ViT_model.onnx.data
```

The model files are downloaded into the `models/` directory when the application starts. They are not required to be committed to GitHub.

The `database.db` file is created locally by the application.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Dharshan225/DRvit.git
cd DRvit
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```cmd
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the Model and Run the Application

The trained model is hosted on Hugging Face:

**Model repository:**  
https://huggingface.co/dharshan25/drvit_model

The model requires these two files:

- `DR_ViT_model.onnx`
- `DR_ViT_model.onnx.data`

The second file contains external model weights required by the ONNX model.

To download the files manually, run:

```bash
python download_model.py
```

The downloader saves the model files to the `models/` directory.

An internet connection is required when the model files are not already available locally.

## Running the Application

Start the Flask application:

```bash
python app.py
```

The application downloads missing model files and initializes ONNX Runtime.

Once the server starts, open:

http://127.0.0.1:5000

Use the web interface to upload a retinal fundus image and view the prediction.

## Screenshots

Screenshots of the application interface, dashboard, and prediction results can be added here.

## Limitations

- This project is intended for educational and research purposes.
- The model's predictions are not a substitute for professional medical evaluation.
- Prediction confidence should not be interpreted as the probability that a diagnosis is correct.
- Performance may vary across datasets, image quality, and patient populations.
- The reported validation accuracy does not establish clinical effectiveness.

Do not use this application as the sole basis for clinical decisions.

## Future Improvements

- Model optimization for edge devices
- Inference speed and memory optimization
- Further evaluation on independent datasets
- Improved model generalization
- Integration with suitable medical imaging workflows

## Author

**Dharshan B**  
Biomedical Engineering Student

GitHub: https://github.com/Dharshan225

## License

A license has not yet been specified. Add an appropriate license before distributing the project.