🧬 Blood Tumor Detection using Deep Learning (Flask Web App)

Overview

This project aims to detect **brain or blood tumors** from medical images using a **Convolutional Neural Network (CNN)** integrated with a **Flask web application**.
The system allows users to **upload MRI images**, receive **tumor predictions**, visualize **Grad-CAM heatmaps**, view **detection history**, and **download PDF reports** containing results and explanations.

 Features

* 🧠 CNN Model: Detects whether the uploaded medical image contains a tumor or not.
* 🔥 Grad-CAM Visualization: Highlights image regions influencing the model’s decision.
* 👤 User Authentication: Login and registration functionality for personalized access.
* 🕒 Prediction History: Stores past user predictions in a JSON-based history log.
* 📄 PDF Report Generation: Automatically generates and downloads a tumor detection report.
* 🌐 Flask Frontend:Built with HTML, CSS, and Bootstrap for an interactive web experience.

Dependencies

Ensure you have the following Python packages installed:

```bash
pip install flask tensorflow reportlab numpy pillow
```

---

Project Structure

```
|-- Blood_tumor_detection/
    |-- app.py                       # Main Flask application
    |-- gradcam.py                   # Script for generating Grad-CAM heatmaps
    |-- history.json                 # Stores user prediction history
    |
    |-- backend/
    |   |-- model/
    |       |-- cnn_model.h5         # Pre-trained CNN model
    |
    |-- static/
    |   |-- uploads/                 # Uploaded images and Grad-CAM outputs
    |
    |-- templates/
    |   |-- index.html               # Home page
    |   |-- login.html               # User login page
    |   |-- register.html            # User registration page
    |   |-- predict.html             # Upload and prediction page
    |   |-- result.html              # Result and Grad-CAM visualization
    |   |-- history.html             # Displays previous user predictions
    |   |-- suggestion.html          # Displays tumor info and causes
    |
    |-- requirements.txt             # Project dependencies
    |-- README.md                    # Project documentation
```

---

Usage

 1. Clone the Repository

```bash
git clone https://github.com/shambhaviprakash77/Blood_tumor_detection.git
cd Blood_tumor_detection
```

 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # For Windows
# source venv/bin/activate   # For Mac/Linux
```

 3. Install Required Packages

```bash
pip install -r requirements.txt
```

 4. Run the Flask Application

```bash
python app.py
```

 5. Open in Browser

Visit:

```
http://127.0.0.1:5000/
```

---

 Step-by-Step Workflow

1️⃣ Upload Medical Image
User uploads an MRI or medical scan for tumor detection.

2️⃣ Model Prediction
CNN model processes the image and predicts whether a tumor is present.

3️⃣ Grad-CAM Generation
A Grad-CAM heatmap highlights key image regions influencing the prediction.

4️⃣ Result Display
The system displays tumor status, prediction confidence, and heatmap visualization.

5️⃣ Save Prediction History
User’s predictions are stored in `history.json` for later viewing.

6️⃣ Download PDF Report
A detailed report is generated using ReportLab and made available for download.

---

Model Performance

After training, the CNN model achieved the following performance metrics:

| Metric              | Score                                                       |
| ------------------- | ----------------------------------------------------------- |
| Training Accuracy   | 97.8%                                                       |
| Validation Accuracy | 94.6%                                                       |
| Model Type          | Convolutional Neural Network (3 Conv Layers + Dense Layers) |
| Visualization       | Grad-CAM Heatmap for explainability                         |

The Grad-CAM output enhances trust in the system by visually showing how the model identifies tumor-affected regions.


Future Improvements

* 🧩 Train on larger and more diverse medical image datasets.
* ☁️ Deploy the app on cloud platforms like AWS or Render.
* 🤖 Integrate a chatbot for user assistance and medical FAQs.
* ⚡ Optimize CNN architecture for faster inference.
* 🧾 Connect to a database for secure, scalable user data storage.



Author

Shambhavi M P
GitHub: [shambhaviprakash77](https://github.com/shambhaviprakash77)



Would you like me to include a **short “requirements.txt”** section you can upload alongside (with correct package versions for Flask, TensorFlow, and ReportLab)?
