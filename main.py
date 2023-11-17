from fastapi import FastAPI, File, UploadFile
from prediction import predict_image, preprocess_image, read_image

app = FastAPI()

@app.get("/")
async def home():
    return {"Message" : "Hello."} 

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):   
    image = read_image(await file.read())
    image = preprocess_image(image)
    img_class = predict_image(image)
    return {"Class": img_class}
