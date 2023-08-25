
import base64
import cv2
import requests
import numpy as np
import pickle
import os
import base64

# ตั้งค่า path สำหรับโฟลเดอร์ข้อมูลภาพรถบน Google Drive ของคุณ
data_path = 'train'
X = []  # เก็บข้อมูลภาพ (แปลงเป็น base64 string)
y = []  # เก็บ label ยี่ห้อรถ

# ใช้ os.listdir() เพื่อดึงรายชื่อไฟล์และโฟลเดอร์ใน path
items = os.listdir(data_path)

# วนลูปในรายชื่อไฟล์และโฟลเดอร์ใน items
for item in items:
    item_path = os.path.join(data_path, item)
    if os.path.isdir(item_path):  # ตรวจสอบว่าเป็นโฟลเดอร์
        for image_file in os.listdir(item_path):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # ตรวจสอบนามสกุลไฟล์ภาพ
                image_path = os.path.join(item_path, image_file)
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # อ่านภาพเป็นสี
                if img is not None:
                    # แปลงรูปภาพเป็น base64 string
                    img_base64 = base64.b64encode(cv2.imencode('.jpg', img)[1].tobytes()).decode('utf-8')
                    X.append(img_base64)  # เพิ่มภาพลงใน X
                    y.append(item)  # เพิ่ม label ยี่ห้อรถ

# แสดงผลลัพธ์
# print("Labels (y):", y)
# print("Number of images (X):", X[1])  # แสดงจำนวนรูปภาพที่ถูกเก็บใน X
url = " http://localhost:80/api/hog"
def img2vec(img):
    # v, buffer = cv2.imencode(".jpg", img)
    # img_str = base64.b64encode(buffer)
    data = "image data,"+str.split(str(img),"'")[1]
    response = requests.post(url, json={"img":data})
    return response.json()
mo=img2vec(X)
# img = cv2.imread('C:\\AI\\train\\Audi\\1.jpg')
# print(mo)
# Write the feature vectors and labels to a pickle file
write_path = "feature_vectors.pkl"
data_to_save = {"X": mo, "y": y}

with open(write_path, "wb") as f:
    pickle.dump(data_to_save, f)

print("Feature vectors and labels saved to", write_path)

# Read the saved feature vectors and labels from a pickle file
read_path = "feature_vectors.pkl"

with open(read_path, "rb") as f:
    loaded_data = pickle.load(f)

X_loaded = loaded_data["X"]
y_loaded = loaded_data["y"]

print("Loaded Feature Vectors:", X_loaded)
print("Loaded Labels:", y_loaded)







# path = 'C:\\AI\\train'
# facvectors = []
# for sub in os.listdir(path):
#     for fn in os.listdir(os.path.join(path,sub)):
#         img_file_name = os.path.join(path,sub)+"/"+fn
#         img = cv2.imread(img_file_name)
#         res = img2vec(img)
#         vec = list(res["vector"])
#         vec.append(int(sub))
#         facvectors.append(vec)
# write_path = "facevectors.pkl"
# pickle.dump(facvectors, open(write_path,"wb"))
# print("data preparation is done")