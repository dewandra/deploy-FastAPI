from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2

app = FastAPI()

MODEL = tf.keras.models.load_model("model.h5")

CLASS_NAMES = ["Bercak", 'Hawar', 'Karat', 'Sehat']

@app.get("/tes")
async def ping():
    return "Hello, from server"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img = cv2.resize(image,(150,150))
    img_batch = np.expand_dims(img, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=80, timeout_keep_alive=1200)
