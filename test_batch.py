import cv2
import numpy as np
import tensorflow as tf
import os
#1 Nạp mô hình và Cấu hình ban đầu
model = tf.keras.models.load_model("emotion_model.h5")

labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

folder = "test_images"
# duyện qua từng ản
for file in os.listdir(folder):
    path = os.path.join(folder, file)
       #2 Tiền xử lý ảnh
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  #Đọc ảnh dưới dạng đen trắng (ảnh xám)
    if img is None:
        continue

    img = cv2.resize(img, (48,48))  #Thu nhỏ/phóng to ảnh về đúng kích thước 48x48
    img = img.astype("float32") / 255.0 # Chuẩn hóa giá trị pixel về khoảng 0-1
    img = img.reshape(1,48,48,1)
  #3 Khối lệnh Dự đoán
    preds = []


    for _ in range(3):     #Đoạn này cho mô hình dự đoán bức ảnh 3 lần và lưu xác suất vào danh sách preds
        p = model.predict(img, verbose=0)[0]
        preds.append(p)

    avg_pred = np.mean(preds, axis=0)   #Tính trung bình cộng xác suất của 3 lần dự đoán
  #4 Giải mã kết quả
    idx = np.argmax(avg_pred)  #Tìm ra vị trí (index) của con số lớn nhất trong 7 số đó
    emotion = labels[idx]         #Dùng vị trí vừa tìm được để tra ngược lại mảng labels
    confidence = np.max(avg_pred) * 100

    print(f"{file} => {emotion} ({confidence:.2f}%)")