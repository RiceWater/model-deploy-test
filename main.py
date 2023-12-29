from fastapi import FastAPI, File, UploadFile
from fastapi import HTTPException
from prediction import check_version, predict_image, preprocess_image, read_image, test

app = FastAPI()

@app.get("/")
async def home():
    return {"Message" : "Hello."} 

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):   
    try:
        image = read_image(await file.read())
        image = preprocess_image(image)
        img_class = predict_image(image)
        return {"Class": img_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/getver/")
async def version():
    v = check_version()
    return {"Version" : v}

@app.get("/test/")
async def testing():
    t = test()
    return {"Version" : t}

@app.post("/checkver/")
async def create_check_ver(file: UploadFile = File(...)): 
    v = check_version()
    return {"Version" : v}
