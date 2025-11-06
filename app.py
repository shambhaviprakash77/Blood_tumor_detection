from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gradcam import generate_gradcam, save_gradcam
from reportlab.pdfgen import canvas
import numpy as np
import os
import uuid
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['HISTORY_FILE'] = 'history.json'

users = {
    "admin@example.com": "password123"
}

model = load_model('backend/model/cnn_model.h5')

def save_history_entry(user_email, entry):
    if not os.path.exists(app.config['HISTORY_FILE']):
        with open(app.config['HISTORY_FILE'], 'w') as f:
            json.dump({}, f)

    with open(app.config['HISTORY_FILE'], 'r+') as f:
        data = json.load(f)
        user_history = data.get(user_email, [])
        user_history.append(entry)
        data[user_email] = user_history
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get("email")
        password = request.form.get("password")
        if users.get(email) == password:
            session['user'] = email
            flash("✅ Login successful!", "success")
            return redirect(url_for("upload"))
        else:
            flash("❌ Invalid credentials. Please try again.", "error")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users:
            flash("⚠️ Email already exists.", "error")
        else:
            users[email] = password
            flash("✅ Registered Successfully! Please login.", "success")
            return redirect(url_for('login'))
    return render_template("register.html")

@app.route('/upload')
def upload():
    if 'user' not in session:
        flash("🔐 Please login to continue", "error")
        return redirect(url_for('login'))
    return render_template("predict.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        flash("🔐 Please login to continue", "error")
        return redirect(url_for('login'))

    patient_name = request.form.get('patient_name')
    file = request.files.get('image')
    if file:
        filename = str(uuid.uuid4()) + ".png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        label = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor"
        confidence = float(prediction[0][0]) if prediction[0][0] > 0.5 else 1 - float(prediction[0][0])

        heatmap = generate_gradcam(model, img_array)
        gradcam_filename = "gradcam_" + filename
        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
        save_gradcam(heatmap, filepath, gradcam_path)

        entry = {
            'patient_name': patient_name,
            'filename': filename,
            'prediction': label,
            'confidence': f"{confidence:.2%}",
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'gradcam': gradcam_filename
        }
        save_history_entry(session['user'], entry)

        return render_template("result.html",
                               patient_name=patient_name,
                               prediction=label,
                               confidence=confidence,
                               img_path=filepath.replace("\\", "/"),
                               gradcam_path=gradcam_path.replace("\\", "/"))
    flash("⚠️ No file uploaded. Please try again.", "error")
    return redirect(url_for('upload'))

@app.route('/history')
def history():
    if 'user' not in session:
        flash("Please login to view history", "error")
        return redirect(url_for('login'))

    user_email = session['user']
    if os.path.exists(app.config['HISTORY_FILE']):
        with open(app.config['HISTORY_FILE'], 'r') as f:
            data = json.load(f)
            user_history = data.get(user_email, [])
    else:
        user_history = []

    return render_template("history.html", history=user_history)

@app.route('/download_report', methods=['POST'])
def download_report():
    patient_name = request.form.get("patient_name", "N/A")
    prediction = request.form.get("prediction")
    confidence = request.form.get("confidence")
    img_path = request.form.get("img_path")
    gradcam_path = request.form.get("gradcam_path")

    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'report.pdf')
    c = canvas.Canvas(pdf_path)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, 800, "🧬 Blood Tumor Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Patient Name: {patient_name}")
    c.drawString(50, 755, f"Prediction: {prediction}")
    c.drawString(50, 740, f"Confidence: {float(confidence):.2%}")
    c.drawString(50, 725, f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 700, "Grad-CAM Explanation:")
    c.setFont("Helvetica", 12)
    c.drawString(50, 685, "- Grad-CAM highlights regions that influenced model predictions.")
    c.drawString(50, 670, "- Helps understand the model's focus on image features.")

    c.showPage()
    c.save()

    return f"<h3>✅ Report generated:</h3><a href='/{pdf_path}'>Download Report</a>"

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
