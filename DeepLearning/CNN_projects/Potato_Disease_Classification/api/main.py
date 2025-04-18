from fastapi import FastAPI, File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    'https://localhost',
    'https://localhost:3000'
]
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)











MODEL = tf.keras.models.load_model('../my_model.keras')
#MODEL = tf.keras.layers.TFSMLayer('CNN_projects/Potato_Disease_Classification/saved_models/3/my_model.keras',call_endpoint = 'serving_default')
CLASS_NAMES = ['Early Blight','Late Blight','Healthy']


@app.get('/ping')
async def ping():
    return "Hello , i am alive"




def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image



@app.post("/predict")
async def predict(
    file: UploadFile = File (...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class':predicted_class,
        'confidence':float(confidence)
    }


if __name__ == '__main__':
    uvicorn.run(app,host='localhost', port=8000)