# 📸 EigenFace Master: Hệ Thống Nhận Dạng Khuôn Mặt Tốc Độ Cao

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![Algorithm](https://img.shields.io/badge/Algorithm-PCA%20%2F%20Eigenfaces-orange.svg)

> **"Nhận dạng khuôn mặt nhanh, nhẹ và hiệu quả dựa trên thuật toán kinh điển."**

Dự án này là một triển khai thực tế của thuật toán **Eigenfaces** (Sử dụng phân tích thành phần chính - PCA) để nhận dạng khuôn mặt qua Webcam. Chương trình được xây dựng dựa trên nền tảng lý thuyết từ hai tài liệu khoa học nổi tiếng:
1.  📄 *Face Recognition Using Eigenfaces* (Turk & Pentland).
2.  📚 *Computer Vision: Algorithms and Applications* (Richard Szeliski).

---

## 🌟 Tính Năng Nổi Bật

* **🚀 Siêu Nhẹ & Nhanh:** Không cần GPU rời, chạy mượt mà trên cả laptop cũ nhờ tối ưu hóa thuật toán PCA và xử lý ảnh nhỏ (Scale Down).
* **🧠 Học Tự Động (Auto-Train):** Tự động phát hiện ảnh mới trong thư mục và huấn luyện lại mô hình chỉ trong vài giây.
* **✂️ Tự Động Cắt Mặt (Smart Crop):** Bạn chỉ cần vứt ảnh chụp vào, hệ thống tự động tìm khuôn mặt, cắt, căn chỉnh và cân bằng sáng.
* **💾 Lưu Trữ Mô Hình:** Không mất thời gian train lại mỗi lần khởi động.
* **🎛️ Tinh Chỉnh Real-time:** Điều chỉnh độ nhạy (Threshold) ngay khi đang chạy camera.

---

## 🛠️ Cài Đặt

Chỉ cần cài đặt 3 thư viện Python cơ bản. Mở Terminal hoặc Command Prompt và chạy:

```bash
pip install opencv-python numpy pillow
📂 Cấu Trúc Thư MụcĐể chương trình chạy đúng, hãy sắp xếp thư mục của bạn như sau:PlaintextMyFaceProject/
│
├── facerecognition.py  # <--- File code chính
├── augment_data.py     # <--- File tạo ảnh từ ảnh gốc: lật, thêm sáng, tối, nhiễu hạt
├── README.md         # <--- File hướng dẫn này
│
└── dataset/               # <--- Thư mục chứa ảnh huấn luyện
    ├── Nguyen_Van_A/      # Tên người thứ 1 (Viết liền, không dấu)
    │   ├── img1.jpg
    │   ├── img2.png
    │   └── ...
    ├── Tran_Thi_B/        # Tên người thứ 2
    │   ├── selfie.jpg
    │   └── ...
    └── ...
========================================================================
                          HƯỚNG DẪN SỬ DỤNG
========================================================================

--- BƯỚC 1: CHUẨN BỊ ẢNH ---
- Vào thư mục "dataset".
- Tạo thư mục tên bạn (ví dụ: "Duy_Khuong").
- Chép khoảng 10-20 tấm ảnh rõ mặt vào đó.
- Lời khuyên: Nên chọn ảnh có ánh sáng khác nhau (sáng, tối, ngược sáng) để máy học giỏi hơn.

--- BƯỚC 2: CHẠY CHƯƠNG TRÌNH ---
- Mở Terminal/CMD tại thư mục dự án.
- Gõ lệnh: 
     python augment_data.py
* Chương trình sẽ tạo sinh các ảnh từ ảnh gốc: thêm sáng, chỉnh tối, nhiễu hạt.

     python facerecognition.py

* Lần đầu chạy: Máy sẽ mất vài giây để quét ảnh và tạo file "face_model.npz".
* Các lần sau: Máy bật Camera lên ngay lập tức.

--- BƯỚC 3: ĐIỀU KHIỂN KHI ĐANG CHẠY ---
Khi cửa sổ Camera hiện lên, bạn dùng bàn phím để điều khiển:

[ q ] -> Thoát chương trình.

[ r ] -> Học lại (Retrain): Bấm phím này sau khi bạn vừa copy thêm ảnh mới vào folder.

[ u ] -> Tăng ngưỡng nhận dạng (Up): 
         Dùng khi máy cứ báo "Unknown" dù đó là người quen.
         
[ d ] -> Giảm ngưỡng nhận dạng (Down):
         Dùng khi máy nhận nhầm người lạ thành người quen.

========================================================================
                       GIẢI QUYẾT LỖI THƯỜNG GẶP
========================================================================

1. Máy không hiện khung xanh quanh mặt?
   -> Kiểm tra ánh sáng (đừng ngồi ngược sáng).
   -> Tháo kính râm/khẩu trang.
   -> Ngồi cách camera khoảng 50-70cm.

2. Máy hiện khung xanh nhưng tên là "Unknown"?
   -> Bấm phím 'u' vài lần để tăng độ nhạy.
   -> Hoặc chụp thêm ảnh tại chỗ rồi bấm 'r' để máy học lại.

3. Máy nhận nhầm người lạ?
   -> Bấm phím 'd' để giảm độ nhạy xuống.

👨‍💻 Tác Giả
   Nguyễn Duy Khương
   Hồ Viết Sơn Tùng
   Bùi Thảo Ly
   Trần Ngọc Minh
   Đỗ Thị Ngọc Mai
Mã nguồn mở và miễn phí sử dụng cho mục đích học tập.

========================================================================
                       CHÚC BẠN THÀNH CÔNG!
========================================================================