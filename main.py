from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from pydantic import BaseModel
from ultralytics import YOLO
from werkzeug.utils import secure_filename

import numpy as np 
from PIL import Image

import firebase_admin
from firebase_admin import credentials, storage

import io 
import cv2
import os
import requests

def load_model():
    modelpath = "./model/final.pt"
    model = YOLO(modelpath)
    return model 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


model = load_model()

app = FastAPI()
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

cred = credentials.Certificate("admin_key.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'food-rec-6b763.appspot.com'  # Thay bằng tên bucket của bạn
})
bucket = storage.bucket()



# Function to upload image to Firebase Storage and get the public URL
def upload_image_to_firebase(file, storage_path):
    bucket = storage.bucket()
    blob = bucket.blob(storage_path)
    blob.upload_from_file(file)
    blob.make_public()  # Tạo URL công khai cho ảnh
    return blob.public_url

# Function to download image from URL and convert to OpenCV format
def download_image_from_url(url):
    response = requests.get(url)
    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

class PredictionResponse(BaseModel):
    best_class_name: str
    highest_confidence: float

@app.post('/upload_image')
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse({"error": "No selected file"}, status_code=400)
    if not allowed_file(file.filename):
        return JSONResponse({"error": "Invalid file type"}, status_code=400)
    
    #Secure file name and upload to Firebase
    filename = secure_filename(file.filename)
    storage_path = f"upload/{filename}"
    image_url = upload_image_to_firebase(file.file, storage_path)
    
    #Download image from URL and make prediction
    image = download_image_from_url(image_url)
    results = model(image, conf=0.8, verbose=False)
    
    
    highest_confidence = 0
    best_class_name = ""
    
    for r in results:
        for idx in range(min(5, len(r.probs.top5))):
            class_idx = r.probs.top5[idx]
            class_name = r.names[class_idx]
            confidence = float(r.probs.top5conf[idx])
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_class_name = class_name
    
    highest_confidence_str = str(highest_confidence)
    decimal_part = "0." + highest_confidence_str.split('.')[1]
        
    
    response_body = PredictionResponse(
        best_class_name=best_class_name,
        highest_confidence=highest_confidence
    )
    return response_body

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse({"error": "Invalid request"}, status_code=400)