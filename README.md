========================================================================
       EIGENFACE MASTER - HỆ THỐNG NHẬN DẠNG KHUÔN MẶT TỐC ĐỘ CAO
========================================================================

Dựa trên thuật toán: Eigenfaces (Turk & Pentland) & Sách Computer Vision (Szeliski)
Ngôn ngữ: Python + OpenCV

[ TÍNH NĂNG CHÍNH ]
1. Nhanh & Nhẹ: Chạy mượt trên mọi laptop, không cần Card đồ họa rời.
2. Tự động cắt mặt: Chỉ cần bỏ ảnh chụp vào, máy tự tìm mặt để học.
3. Học lại (Retrain) tức thì: Bấm 1 phím là cập nhật người mới.
4. Lưu Model: Không mất thời gian học lại mỗi lần mở máy.

========================================================================
                               CÀI ĐẶT
========================================================================

Bước 1: Cài đặt Python (nếu chưa có).

Bước 2: Mở Terminal (CMD) và chạy lệnh sau để cài thư viện:
   pip install opencv-python numpy pillow

========================================================================
                          CẤU TRÚC THƯ MỤC
========================================================================
Để phần mềm chạy đúng, bạn hãy sắp xếp thư mục y hệt như sau:

Thu_muc_du_an/
  |
  +-- final_face_rec.py       (File code chính)
  +-- HUONG_DAN_SU_DUNG.txt   (File này)
  |
  +-- dataset/                (THƯ MỤC QUAN TRỌNG NHẤT)
       |
       +-- Nguyen_Van_A/      (Tạo thư mục tên người muốn nhận dạng)
       |     +-- anh1.jpg
       |     +-- anh2.png
       |
       +-- Tran_Thi_B/
             +-- hinh_chup.jpg
             +-- ...

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
     python final_face_rec.py

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

========================================================================
                       CHÚC BẠN THÀNH CÔNG!
========================================================================