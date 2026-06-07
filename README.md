# Dự Án Nhận Diện Cảm Xúc

Dự án này sử dụng mô hình mạng nơ-ron tích chập (CNN - Convolutional Neural Network) được huấn luyện trên thư viện TensorFlow/Keras để phân loại các biểu cảm cảm xúc. Hệ thống kết hợp với OpenCVđể trích xuất hình ảnh trực tiếp từ Webcam, tự động khoanh vùng khuôn mặt và dự đoán cảm xúc thời gian thực.
Mô hình phân loại chính xác 7 trạng thái biểu cảm:
Angry(Tức giận)
Disgust (Ghê tởm)
Fear(Sợ hãi)
Happy(Vui vẻ)
Neutral(Bình thường)
Sad(Buồn bã)
Surprise (Ngạc nhiên)

## 📂 Cấu trúc thư mục dự án

```text
EmotionProject/
│
├── dataset/archive/          # Thư mục chứa tập dữ liệu ảnh huấn luyện (Dataset)
│   ├── train/                # Tập ảnh dùng để train mô hình (7 thư mục cảm xúc)
│   └── test/                 # Tập ảnh dùng để kiểm thử/đánh giá (Validation)
│
├── test_images/              # Thư mục chứa các ảnh tĩnh bên ngoài để test thử nghiệm
│
├── best_emotion_model.keras  # File mô hình lưu ở định dạng Keras mới (Độ chính xác cao nhất)
├── emotion_model.h5          # File mô hình và trọng số định dạng H5 truyền thống
│
├── train_model.py            # Code xây dựng mạng CNN, tiền xử lý ảnh và huấn luyện mô hình
├── test_batch.py             # Code kiểm tra mô hình hàng loạt bằng các ảnh tĩnh trong test
├── test_camera.py            # Code chạy ứng dụng nhận diện biểu cảm trực tiếp qua Webcam
│
├── requirements.txt          # Danh sách các thư viện cần cài đặt cho hệ thống
└── README.md                 # Tài liệu hướng dẫn sử dụng dự án
# hướng dẫn vận hành chi tiết
Lệnh để Train lại mô hình :"python train_model.py"

Lệnh để Test ảnh :"python test_batch.py"

Lệnh để Demo thời gian thực:"python test_camera.py"

nút tắt camera : nhấn "q"
# Yêu cầu hệ thống và Cài đặt môi trường
  "pip install -r requirements.txt"
```

# thư viện sử dụng

TensorFlow & Keras: Xây dựng cấu trúc mạng CNN chuyên sâu (Conv2D, MaxPooling2D, Dropout, Dense), tối ưu hóa thuật toán và tính toán ma trận trọng số.

OpenCV (cv2): Đọc luồng video từ webcam, chuyển đổi hình ảnh sang hệ màu xám (Gray Scale), áp dụng bộ lọc phát hiện khuôn mặt và vẽ khung chữ đồ họa.

NumPy: Tiền xử lý mảng ma trận điểm ảnh (Reshape, Normalize dữ liệu ảnh về khoảng từ 0 đến 1) giúp tăng tốc độ xử lý phần cứng.
