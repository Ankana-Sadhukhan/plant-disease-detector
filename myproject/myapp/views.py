import os
import numpy as np
from PIL import Image
from django.shortcuts import render
from tensorflow.keras.models import load_model
from django.conf import settings

# --- GLOBAL CONFIGURATION ---
# Use os.path.join to ensure the path works on Render's Linux system
MODEL_PATH = os.path.join(settings.BASE_DIR, 'myapp', 'model', 'plant_disease_best.keras')

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight", "Corn___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def predict_disease(request):
    prediction = None
    confidence = None
    error = None

    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # 1. Get the image from the request
            image_file = request.FILES['image']

            # 2. Process Image
            img = Image.open(image_file)
            img = img.resize((128, 128))  # Must match your training size
            img_array = np.array(img) / 255.0
            
            # Handle RGB conversion if image is RGBA
            if img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
                
            img_array = np.expand_dims(img_array, axis=0)

            # 3. LAZY LOADING: Load model only when needed to save RAM
            # Using MODEL_PATH (absolute path) instead of relative string
            model = load_model(MODEL_PATH)

            # 4. Predict
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions)
            
            # 5. Format Result
            raw_result = CLASS_NAMES[predicted_index]
            prediction = raw_result.replace("___", " - ").replace("_", " ")
            confidence = round(float(np.max(predictions)) * 100, 2)
            
            print(f"SUCCESS: Predicted {prediction} with {confidence}% confidence")

        except Exception as e:
            print(f"DEPLOYMENT ERROR: {e}")
            error = "Model loading failed. The file may be too large for the free tier RAM."

    return render(request, 'index.html', {
        'prediction': prediction,
        'confidence': confidence,
        'error': error
    })