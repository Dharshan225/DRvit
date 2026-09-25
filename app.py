from flask import Flask, render_template, request
import os
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename

from predict import predict_image


app = Flask(__name__)

# =========================================================
# CONFIGURATION
# =========================================================

UPLOAD_FOLDER = "static/uploads"
DATABASE = "database.db"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# DATABASE
# =========================================================

def init_db():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
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

    conn.commit()
    conn.close()


init_db()


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

    # Check image
    if "image" not in request.files:

        return "No Image Uploaded"


    file = request.files["image"]


    if file.filename == "":

        return "No Image Selected"


    # -----------------------------------------------------
    # GET PATIENT DETAILS
    # -----------------------------------------------------

    patient_id = request.form.get(
        "patient_id",
        ""
    ).strip()

    patient_name = request.form.get(
        "patient_name",
        ""
    ).strip()

    age = request.form.get(
        "age",
        ""
    ).strip()

    gender = request.form.get(
        "gender",
        ""
    ).strip()

    contact = request.form.get(
        "contact",
        ""
    ).strip()


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if not patient_id:

        return "Patient ID is required"


    if not patient_name:

        return "Patient Name is required"


    if not age:

        return "Patient Age is required"


    if not gender:

        return "Gender is required"


    try:

        age = int(age)

    except ValueError:

        return "Age must be a number"


    # -----------------------------------------------------
    # SAVE IMAGE
    # -----------------------------------------------------

    filename = secure_filename(
        file.filename
    )


    if not filename:

        return "Invalid image file"


    # Add timestamp to prevent duplicate filenames

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    name, extension = os.path.splitext(
        filename
    )


    filename = (
        f"{patient_id}_{timestamp}{extension}"
    )


    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    file.save(filepath)


    # -----------------------------------------------------
    # AI PREDICTION
    # -----------------------------------------------------

    try:

        result = predict_image(filepath)

        prediction = result["prediction"]

        confidence = float(
            result["confidence"]
        )

    except Exception as e:

        return f"Prediction Error: {str(e)}"


    # -----------------------------------------------------
    # DATE AND TIME
    # -----------------------------------------------------

    current_datetime = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    # -----------------------------------------------------
    # SAVE PATIENT + RESULT TO DATABASE
    # -----------------------------------------------------

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()


    cursor.execute("""
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


    conn.commit()

    conn.close()


    # -----------------------------------------------------
    # SHOW RESULT
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
        "patient_id",
        ""
    ).strip()


    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()


    # -----------------------------------------------------
    # TOTAL SCREENINGS
    # -----------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM patients
    """)

    total_screenings = cursor.fetchone()[0]


    # -----------------------------------------------------
    # TOTAL UNIQUE PATIENTS
    # -----------------------------------------------------

    cursor.execute("""
        SELECT COUNT(DISTINCT patient_id)
        FROM patients
    """)

    total_patients = cursor.fetchone()[0]


    # -----------------------------------------------------
    # TODAY'S SCREENINGS
    # -----------------------------------------------------

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )


    cursor.execute("""
        SELECT COUNT(*)
        FROM patients
        WHERE date LIKE ?
    """, (
        today + "%",
    ))


    today_screenings = cursor.fetchone()[0]


    # -----------------------------------------------------
    # RECENT RECORDS
    # -----------------------------------------------------

    cursor.execute("""
        SELECT *
        FROM patients
        ORDER BY id DESC
        LIMIT 10
    """)


    recent_patients = cursor.fetchall()


    # -----------------------------------------------------
    # SEARCH PATIENT
    # -----------------------------------------------------

    search_results = []


    if search_id:

        cursor.execute("""
            SELECT *
            FROM patients
            WHERE patient_id LIKE ?
            ORDER BY id DESC
        """, (
            "%" + search_id + "%",
        ))


        search_results = cursor.fetchall()


    conn.close()


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
# PATIENT HISTORY
# =========================================================

@app.route("/patient/<patient_id>")
def patient_history(patient_id):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()


    cursor.execute("""
        SELECT *
        FROM patients
        WHERE patient_id = ?
        ORDER BY id DESC
    """, (
        patient_id,
    ))


    history = cursor.fetchall()


    conn.close()


    return render_template(

        "patient_history.html",

        history=history

    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(debug=True)