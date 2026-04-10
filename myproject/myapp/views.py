from django.shortcuts import render

# Create your views here.
import os
import numpy as np
from PIL import Image
from django.shortcuts import render
from tensorflow.keras.models import load_model
from django.conf import settings

# Load model only once for fast prediction
model = load_model('myapp/model/plant_disease_best.keras')

# class_names = [
#     "Healthy",
#     "Bacterial Spot",
#     "Leaf Mold",
#     "Early Blight",
#     "Late Blight"
# ]

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def predict_disease(request):
    prediction = None
    confidence = None

    if request.method == 'POST':
        image = request.FILES['image']

        img = Image.open(image)
        img = img.resize((128, 128))  # use your training size
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        # print(pred)
        # print(pred.shape)
        # print(np.argmax(pred))
        # print(len(class_names))
        result = class_names[np.argmax(pred)]

        prediction = result
        confidence = round(float(np.max(pred)) * 100, 2)
        print("MODEL EXECUTED SUCCESSFULLY")
        # print(pred)
        # print(np.argmax(pred))

    return render(request, 'index.html', {'prediction': prediction,"confidence": confidence})

# prediction = class_names[predicted_index].replace("___", " - ")
    

# return render(request, "index.html", {
#     "prediction": prediction,
    
# })