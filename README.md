# 🌿 Plant Disease Detector

An AI-powered web application that detects plant diseases from leaf images using Deep Learning and Computer Vision techniques. This project helps farmers, researchers, and agriculture enthusiasts identify plant diseases quickly and efficiently for early treatment and improved crop health.

---

## 🚀 Features

* 📸 Upload plant leaf images for disease prediction
* 🤖 Deep Learning based disease classification
* ⚡ Fast and accurate predictions
* 🌱 Supports multiple plant disease categories
* 🖥️ Simple and user-friendly interface
* 📊 Real-time prediction results
* ☁️ Easy deployment and scalability

---

## 🧠 Tech Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend / ML

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Flask / Streamlit *(depending on your implementation)*

### Dataset

* PlantVillage Dataset

---

## 📂 Project Structure

```bash
plant-disease-detector/
│
├── dataset/               # Training dataset
├── model/                 # Trained ML models
├── static/                # CSS, JS, Images
├── templates/             # HTML templates
├── app.py                 # Main application file
├── train_model.py         # Model training script
├── requirements.txt       # Required dependencies
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ankana-Sadhukhan/plant-disease-detector.git
cd plant-disease-detector
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### 3️⃣ Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python app.py
```

After running the command, open your browser and go to:

```bash
http://127.0.0.1:5000/
```

---

## 🖼️ How It Works

1. User uploads a leaf image
2. Image preprocessing is performed
3. The trained CNN model analyzes the image
4. Disease prediction is generated
5. Result is displayed with confidence score

---

## 📊 Model Information

This project uses a Convolutional Neural Network (CNN) trained on plant leaf datasets to classify diseases based on image patterns and symptoms.

Typical pipeline includes:

* Image preprocessing
* Data augmentation
* CNN training
* Model evaluation
* Prediction generation

Deep Learning based plant disease detection has shown strong performance in agricultural applications. ([arXiv][1])

---

## 🌱 Supported Disease Categories

Example categories may include:

* Tomato Early Blight
* Tomato Late Blight
* Potato Healthy
* Pepper Bell Bacterial Spot
* Corn Common Rust
* Healthy Leaves

*(Modify according to your trained dataset)*

---

## 📸 Screenshots

### Home Page

*Add project screenshots here*

### Prediction Result

*Add prediction result screenshots here*

---

## 🔮 Future Improvements

* Mobile application integration
* Real-time camera detection
* Multi-language support
* Disease treatment recommendation system
* Cloud deployment
* Farmer dashboard and analytics

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to the branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 👩‍💻 Author

Developed by Ankana Sadhukhan

GitHub: [Ankana-Sadhukhan GitHub Profile](https://github.com/Ankana-Sadhukhan?utm_source=chatgpt.com)

Repository: [Plant Disease Detector Repository](https://github.com/Ankana-Sadhukhan/plant-disease-detector?utm_source=chatgpt.com)

---

## ⭐ Support

If you like this project, please consider giving it a ⭐ on GitHub.

[1]: https://arxiv.org/abs/2003.05379?utm_source=chatgpt.com "Plant Disease Detection from Images"
