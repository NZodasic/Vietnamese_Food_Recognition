from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware


from sqlalchemy.orm import Session
from jose import JWTError, jwt

from datetime import datetime, timedelta
from passlib.context import CryptContext

from models import User
from database import SessionLocal, engine

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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def load_model():
    modelpath = "./model/final.pt"
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


model = load_model()

app = FastAPI()


origins = [
    "http://localhost:3000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,   # Allow cookies and other credentials
    allow_methods=["*"],      # Allow all HTTP methods (or specify which ones you need)
    allow_headers=["*"],      # Allow all headers (or specify which ones you need)
)



ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "65a8e27d8879283831b664bd8b7f0ad4"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class UserCreate(BaseModel):
    username: str
    password: str


def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = pwd_context.hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    return "complete"

# Authenticate the user
def authenticate_user(username: str, password: str, db: Session):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not pwd_context.verify(password, user.hashed_password):
        return False
    return user

# Create access token
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

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

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=403, detail="Token is invalid or expired")
        return payload
    except JWTError:
        raise HTTPException(status_code=403, detail="Token is invalid or expired")

class PredictionResponse(BaseModel):
    best_class_name: str
    highest_confidence: float

@app.post("/")
def home():
    return {"message": "Welcome to the API"}
@app.post("/register")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    return create_user(db=db, user=user)

@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/verify-token/{token}")
async def verify_user_token(token: str):
    verify_token(token=token)
    return {"message": "Token is valid"}

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
