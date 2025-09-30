# app.py
from fastapi import FastAPI, UploadFile, File
from inference import predict

app = FastAPI(title="Brain Tumor Classification API")

@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    # Save uploaded file
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Run prediction
    result = predict(file_location)
    return result

@app.get("/")
def read_root():
    return {"message": "Welcome to Brain Tumor Classification API"}
