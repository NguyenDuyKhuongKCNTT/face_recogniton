# 📸 Real-Time Face Recognition System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![DeepFace](https://img.shields.io/badge/AI-DeepFace-orange)
![MediaPipe](https://img.shields.io/badge/Detection-MediaPipe-green)
![Norfair](https://img.shields.io/badge/Tracking-Norfair-red)

Hệ thống nhận diện khuôn mặt thời gian thực (Real-time) được xây dựng bằng Python. Dự án kết hợp sức mạnh của **MediaPipe** (tốc độ phát hiện cao), **DeepFace** (nhận diện chính xác) và **Norfair** (theo dõi đối tượng ổn định).

## 🚀 Tính năng nổi bật

* **Tốc độ cao:** Sử dụng MediaPipe để phát hiện khuôn mặt, hoạt động mượt mà trên CPU (không cần GPU rời).
* **Ổn định (Anti-Flicker):** Tích hợp thuật toán Tracking (Norfair) giúp gán ID cho từng người, tránh hiện tượng tên bị nhấp nháy hoặc mất dấu khi quay mặt.
* **Đa luồng (Multi-threading):** Quá trình nhận diện AI chạy song song với luồng hiển thị Video, giúp FPS luôn ổn định, không bị giật lag.
* **Dễ dàng cài đặt:** Không yêu cầu C++ Build Tools phức tạp, chỉ cần Python thuần.

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ / Thư viện | Vai trò |
| :--- | :--- | :--- |
| **Language** | Python | Ngôn ngữ lập trình chính |
| **Detector** | MediaPipe | Phát hiện vị trí khuôn mặt trong khung hình |
| **Recognizer** | DeepFace (VGG-Face) | Trích xuất đặc trưng và so khớp khuôn mặt |
| **Tracker** | Norfair | Theo dõi vị trí (ID) của khuôn mặt qua các frame |
| **Interface** | OpenCV | Xử lý hình ảnh và hiển thị Webcam |

## 📂 Cấu trúc dự án

```text
Face-Recognition-System/
├── db/                     # Cơ sở dữ liệu ảnh khuôn mặt
│   ├── Person_Name_1/      # Thư mục chứa ảnh của người 1
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── Person_Name_2/      # Thư mục chứa ảnh của người 2
│   └── ...
├── main.py                 # Mã nguồn chính của chương trình
├── requirements.txt        # Danh sách các thư viện cần thiết
└── README.md               # Tài liệu hướng dẫn
⚙️ Hướng dẫn cài đặt
1. Yêu cầu tiên quyết
Python 3.8 hoặc mới hơn.

Webcam.

2. Cài đặt thư viện
Mở Terminal tại thư mục dự án và chạy lệnh sau để cài đặt tất cả các thư viện phụ thuộc:

Bash

pip install -r requirements.txt
Nội dung file requirements.txt:

Plaintext

numpy
opencv-python
deepface
tf-keras
mediapipe
norfair
🖥️ Hướng dẫn sử dụng
Bước 1: Chuẩn bị dữ liệu
Tạo các thư mục con trong thư mục db/ tương ứng với tên của người cần nhận diện. Chép hình ảnh chân dung (rõ mặt) vào các thư mục đó.

Ví dụ:

db/NguyenVanA/anh1.jpg

db/LeThiB/anh_chan_dung.png

Bước 2: Chạy chương trình
Chạy lệnh sau trong Terminal:

Bash

python main.py
Bước 3: Trải nghiệm
Hệ thống sẽ tự động bật Webcam.

Lần chạy đầu tiên có thể mất vài phút để tải Model (VGG-Face).

Nhấn phím 'q' để thoát chương trình.

🤝 Đóng góp
Dự án được xây dựng với mục đích học tập và nghiên cứu. Mọi đóng góp (Pull Requests) để cải thiện hiệu năng hoặc thêm tính năng mới (như Liveness Detection) đều được hoan nghênh.

Author: [Nhóm 2 Thị Giác Máy Tính - Nguyễn Duy Khương - Hồ Viết Sơn Tùng - Đỗ Thị Ngọc Mai - Bùi Thảo Ly - Trần Ngọc Minh]