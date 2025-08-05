# from flask import Flask, render_template, request
# import cv2
# import numpy as np
# import os
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load the trained model

# MODEL_PATH = r"C:\PYTHON_PROGRAMMING\MACHINE_LEARNING\Update_project\Breast_cancer_detection2.py\breast_cancer_vgg16.keras"
# # MODEL_PATH = "breast_cancer_vgg16.keras"



# model = load_model(MODEL_PATH)

# # Define upload folder
# UPLOAD_FOLDER = "static/uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# # Allowed file extensions
# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}

# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (224, 224))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#     image = image / 255.0  # Normalize
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# @app.route("/", methods=["GET", "POST"])
# def home():
#     if request.method == "POST":
#         try:
#             file = request.files["file"]
#             if file and allowed_file(file.filename):
#                 file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#                 file.save(file_path)
                
#                 image = preprocess_image(file_path)
#                 prediction = model.predict(image)
#                 confidence = float(np.max(prediction)) * 100
#                 result = ["Benign", "Malignant"][np.argmax(prediction)]
                
#                 return render_template(
#                     "index.html",
#                     prediction=result,
#                     confidence=f"{confidence:.2f}",
#                     image_file=file.filename
#                 )
#         except Exception as e:
#             print(f"Error: {e}")
#             return render_template(
#                 "index.html",
#                 prediction="Error processing image",
#                 confidence="0.00",
#                 image_file=None
#             )
    
#     return render_template("index.html", prediction=None, confidence=None, image_file=None)


# if __name__ == "__main__":
#     app.run(debug=True)


# THIS CODE RUN IN CMD 

# from flask import Flask, render_template, request
# from werkzeug.utils import secure_filename
# import cv2
# import numpy as np
# import os
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Option A: If model is in same directory as app.py
# MODEL_PATH = "breast_cancer_vgg16.keras"


# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found at: {os.path.abspath(MODEL_PATH)}")

# model = load_model(MODEL_PATH)

# # 2. FIX UPLOAD FOLDER CREATION
# UPLOAD_FOLDER = "static/uploads"

# # Robust folder creation
# try:
#     if os.path.exists(UPLOAD_FOLDER) and not os.path.isdir(UPLOAD_FOLDER):
#         os.remove(UPLOAD_FOLDER)  # Remove if it's a file
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# except Exception as e:
#     print(f"Error creating upload folder: {e}")
#     raise

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}

# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Could not read image file")
#     image = cv2.resize(image, (224, 224))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = image / 255.0
#     return np.expand_dims(image, axis=0)

# @app.route("/", methods=["GET", "POST"])
# def home():
#     if request.method == "POST":
#         if "file" not in request.files:
#             return render_template("index.html", 
#                                 prediction="No file selected",
#                                 confidence="0.00",
#                                 image_file=None)
            
#         file = request.files["file"]
#         if file.filename == "":
#             return render_template("index.html",
#                                 prediction="No file selected",
#                                 confidence="0.00",
#                                 image_file=None)
            
#         if file and allowed_file(file.filename):
#             try:
#                 filename = secure_filename(file.filename)
#                 file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#                 file.save(file_path)
                
#                 image = preprocess_image(file_path)
#                 prediction = model.predict(image)
#                 confidence = float(np.max(prediction)) * 100
#                 result = "Benign" if np.argmax(prediction) == 0 else "Malignant"
                
#                 return render_template(
#                     "index.html",
#                     prediction=result,
#                     confidence=f"{confidence:.2f}%",
#                     image_file=filename
#                 )
#             except Exception as e:
#                 print(f"Error processing file: {e}")
#                 return render_template(
#                     "index.html",
#                     prediction="Error processing image",
#                     confidence="0.00",
#                     image_file=None
#                 )
    
#     return render_template("index.html", prediction=None, confidence=None, image_file=None)

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000, debug=True)



# print(secrets.token_hex(16))  
import secrets
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO

app = Flask(__name__)
app.secret_key = '3efd4891afe8ed7f59138ba184245dc6'  # Replace with a secure key

# Dummy login credentials
USERNAME = 'Mannan'
PASSWORD = 'Mannan123'

# Load the trained model
MODEL_PATH = "breast_cancer_vgg16.keras"
model = load_model(MODEL_PATH)

# Upload folder setup
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Allowed file types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image file")
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

@app.route("/")
def root():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == USERNAME and password == PASSWORD:
            session["user"] = username
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/home", methods=["GET", "POST"])
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files.get("file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            try:
                image = preprocess_image(file_path)
                prediction = model.predict(image)
                confidence = float(np.max(prediction)) * 100
                result = "Benign" if np.argmax(prediction) == 0 else "Malignant"

                # Store result in session for download access
                session['prediction'] = result
                session['confidence'] = f"{confidence:.2f}"
                session['image_file'] = filename

                return render_template("index.html", prediction=result, confidence=f"{confidence:.2f}", image_file=filename)
            except Exception as e:
                return render_template("index.html", prediction="Error processing image", confidence="0.00", image_file=None)

    return render_template("index.html", prediction=session.get('prediction'), confidence=session.get('confidence'), image_file=session.get('image_file'))

@app.route("/download_form", methods=["GET", "POST"])
def download_form():
    if "prediction" not in session or "confidence" not in session:
        return redirect(url_for("home"))

    return render_template("download.html")


@app.route("/generate_report", methods=["POST"])
def generate_report():
    if "prediction" not in session or "confidence" not in session:
        return redirect(url_for("home"))

    patient_name = request.form.get("patient_name", "Unknown")
    patient_age = request.form.get("patient_age", "Unknown")
    
    from datetime import datetime
    today = datetime.today().strftime("%Y-%m-%d")

    result_text = (
        "==== BREAST CANCER DETECTION REPORT ====\n"
        "----------------------------------------\n"
        f"Patient Name      : {patient_name}\n"
        f"Patient Age       : {patient_age}\n"
        f"Date              : {today}\n\n"
        f"Diagnosis         : {session['prediction']}\n"
        f"Confidence Level  : {session['confidence']}%\n"
        "----------------------------------------\n"
       
    )

    buffer = BytesIO()
    buffer.write(result_text.encode('utf-8'))
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="breast_cancer_report.txt", mimetype="text/plain")


@app.route('/performance')
def performance():
    return render_template('performance.html')

if __name__ == "__main__":
    app.run(debug=True)
