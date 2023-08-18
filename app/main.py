from fastapi import FastAPI, Request

import pickle
import numpy as np
import cv2
import base64
from hog import hog_des

app = FastAPI()

def base642img(str_img):
   encoded_data = str_img.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img
  
@app.get("/")
def root():
        return {"message": "This is my api"}

@app.get("/api/hog")
async def read_str(data : Request):
    json = await data.json()
    item_str = json['img']
    img = base642img(item_str)
    hog = hog_des(img)
    return {'hog':hog.tolist()}