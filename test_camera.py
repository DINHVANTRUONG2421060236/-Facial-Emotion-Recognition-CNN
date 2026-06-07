import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
#1Khởi tạo Mô hình và Công cụ
model = tf.keras.models.load_model("emotion_model.h5")
labels = ['angry','disgust','fear','happy','neutral','sad','surprise']
    # Tải bộ dò tìm khuôn mặt có sẵn của OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
   #Mở Webcam
cap = cv2.VideoCapture(0)
# 2 Cấu hình bộ đệm Làm mượt
buffer = deque(maxlen=7)

last_time = 0
emotion = "Detecting..."
#3 Vòng lặp Video và Tìm khuôn mặt
while True:
    ret, frame = cap.read()     ## Đọc từng khung hình từ Webcam
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Chuyển khung hình sang ảnh xám để tìm khuôn mặt cho nhanh

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]   #Cắt riêng phần khuôn mặt ra khỏi khung hình lớn
#4 Tiền xử lý và AI Dự đoán
        try:
            face = cv2.resize(face, (48,48))
        except:
            continue

        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0,-1))

        pred = model.predict(face, verbose=0)[0] # AI dự đoán cảm xúc của khuôn mặt bị cắt

        buffer.append(pred)

  #5 Logic Cập nhật Giao diện (Mỗi 0.5s)      
        if time.time() - last_time > 0.5:
            avg_pred = np.mean(buffer, axis=0)

            idx = np.argmax(avg_pred)
            emotion = labels[idx]
            confidence = np.max(avg_pred) * 100

            last_time = time.time()

  #6 Vẽ Giao diện và Hiển thị  
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        text = f"{emotion} {confidence:.1f}%" if buffer else "Detecting..."
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0), 2)

    cv2.imshow("Emotion AI (Smooth)", frame)
       ## Nếu nhấn phím 'q'thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Tắt Webcam và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()