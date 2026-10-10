"""
DR ViT - Diabetic Retinopathy Screening Application

Features:
- Retinal image upload
- ONNX model inference
- Patient information management
- SQLite database storage
- Screening dashboard
- Patient search and screening history
- Hugging Face model downloading when files are missing
"""

import os
import sqlite3

from datetime import datetime
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

DATABASE = os.path.join(
    BASE_DIR,
    "database.db"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

# Create required directories.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# ENSURE MODEL FILES ARE AVAILABLE
# =========================================================

# Download missing model files before importing predict.py.
# predict.py imports model.py, which loads the ONNX model.

from download_model import download_model

download_model()


# Import the prediction function only after checking the model.
from predict import predict_image


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Maximum upload size: 10 MB.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_db_connection():
    """Create a connection to the SQLite database."""

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    return conn


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def init_db():
    """Create the patient table if it does not exist."""

    with get_db_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                contact TEXT,
                image TEXT,
                prediction TEXT,
                confidence REAL,
                date TEXT
            )
        """)


init_db()


# =========================================================
# FILE VALIDATION
# =========================================================

def allowed_file(filename):
    """Check whether the uploaded file has an allowed extension."""

    if not filename or "." not in filename:
        return False

    extension = os.path.splitext(filename)[1].lower()

    return extension in ALLOWED_EXTENSIONS


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------------------------------
    # CHECK IMAGE UPLOAD
    # -----------------------------------------------------

    if "image" not in request.files:
        return "No image uploaded.", 400

    file = request.files["image"]

    if not file or file.filename == "":
        return "No image selected.", 400

    if not allowed_file(file.filename):
        return (
            "Unsupported image format. "
            "Upload JPG, JPEG, PNG, BMP, TIF, or TIFF.",
            400
        )

    # -----------------------------------------------------
    # GET PATIENT DETAILS
    # -----------------------------------------------------

    patient_id = request.form.get(
        "patient_id", ""
    ).strip()

    patient_name = request.form.get(
        "patient_name", ""
    ).strip()

    age_input = request.form.get(
        "age", ""
    ).strip()

    gender = request.form.get(
        "gender", ""
    ).strip()

    contact = request.form.get(
        "contact", ""
    ).strip()

    # -----------------------------------------------------
    # VALIDATE PATIENT DETAILS
    # -----------------------------------------------------

    if not patient_id:
        return "Patient ID is required.", 400

    if not patient_name:
        return "Patient name is required.", 400

    if not age_input:
        return "Patient age is required.", 400

    if not gender:
        return "Gender is required.", 400

    try:
        age = int(age_input)

    except ValueError:
        return "Age must be a valid number.", 400

    if age < 0 or age > 125:
        return "Enter a valid age.", 400

    # -----------------------------------------------------
    # PREPARE UPLOADED IMAGE
    # -----------------------------------------------------

    original_filename = secure_filename(file.filename)

    if not original_filename:
        return "Invalid image filename.", 400

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    # Timestamp plus a unique suffix prevents collisions.
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    safe_patient_id = secure_filename(patient_id)

    if not safe_patient_id:
        return "Invalid patient ID.", 400

    filename = f"{safe_patient_id}_{timestamp}{extension}"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # -----------------------------------------------------
    # SAVE IMAGE
    # -----------------------------------------------------

    try:
        file.save(filepath)

    except Exception:
        app.logger.exception("Failed to save uploaded image.")
        return "Could not save the uploaded image.", 500

    # -----------------------------------------------------
    # RUN DR VI T PREDICTION
    # -----------------------------------------------------

    try:
        result = predict_image(filepath)

        prediction = result["prediction"]
        confidence = float(result["confidence"])

    except Exception:
        # Log technical details on the server rather than
        # returning internal exception details to the browser.
        app.logger.exception("DR ViT prediction failed.")

        # Remove the image if prediction fails.
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            app.logger.warning(
                "Could not remove the failed upload: %s",
                filepath
            )

        return (
            "Prediction failed. Check the application terminal "
            "for the error details.",
            500
        )

    # -----------------------------------------------------
    # CURRENT DATE AND TIME
    # -----------------------------------------------------

    current_datetime = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # -----------------------------------------------------
    # SAVE PATIENT AND PREDICTION TO DATABASE
    # -----------------------------------------------------

    try:
        with get_db_connection() as conn:

            conn.execute("""
                INSERT INTO patients (
                    patient_id,
                    patient_name,
                    age,
                    gender,
                    contact,
                    image,
                    prediction,
                    confidence,
                    date
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id,
                patient_name,
                age,
                gender,
                contact,
                filename,
                prediction,
                confidence,
                current_datetime
            ))

    except sqlite3.Error:
        app.logger.exception(
            "Failed to save the screening record."
        )

        return (
            "The prediction completed, but the record "
            "could not be saved.",
            500
        )

    # -----------------------------------------------------
    # DISPLAY PREDICTION RESULT
    # -----------------------------------------------------

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image=filename,
        patient_id=patient_id,
        patient_name=patient_name,
        age=age,
        gender=gender,
        contact=contact
    )


# =========================================================
# PATIENT DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    search_id = request.args.get(
        "patient_id", ""
    ).strip()

    # -----------------------------------------------------
    # FETCH DASHBOARD STATISTICS
    # -----------------------------------------------------

    with get_db_connection() as conn:

        total_screenings = conn.execute("""
            SELECT COUNT(*)
            FROM patients
        """).fetchone()[0]

        total_patients = conn.execute("""
            SELECT COUNT(DISTINCT patient_id)
            FROM patients
        """).fetchone()[0]

        today = datetime.now().strftime("%Y-%m-%d")

        today_screenings = conn.execute("""
            SELECT COUNT(*)
            FROM patients
            WHERE date LIKE ?
        """, (
            today + "%",
        )).fetchone()[0]

        # -------------------------------------------------
        # RECENT SCREENING RECORDS
        # -------------------------------------------------

        recent_patients = conn.execute("""
            SELECT *
            FROM patients
            ORDER BY id DESC
            LIMIT 10
        """).fetchall()

        # -------------------------------------------------
        # SEARCH PATIENT
        # -------------------------------------------------

        search_results = []

        if search_id:

            search_results = conn.execute("""
                SELECT *
                FROM patients
                WHERE patient_id LIKE ?
                ORDER BY id DESC
            """, (
                "%" + search_id + "%",
            )).fetchall()

    # -----------------------------------------------------
    # RENDER DASHBOARD
    # -----------------------------------------------------

    return render_template(
        "dashboard.html",
        total_patients=total_patients,
        total_screenings=total_screenings,
        today_screenings=today_screenings,
        recent_patients=recent_patients,
        search_results=search_results,
        search_id=search_id
    )


# =========================================================
# PATIENT SCREENING HISTORY
# =========================================================

@app.route("/patient/<patient_id>")
def patient_history(patient_id):

    with get_db_connection() as conn:

        history = conn.execute("""
            SELECT *
            FROM patients
            WHERE patient_id = ?
            ORDER BY id DESC
        """, (
            patient_id,
        )).fetchall()

    return render_template(
        "patient_history.html",
        history=history
    )


# =========================================================
# UPLOAD SIZE ERROR HANDLER
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return "Image too large. Maximum upload size is 10 MB.", 413


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )