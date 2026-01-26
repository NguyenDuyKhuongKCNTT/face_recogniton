import cv2
import numpy as np
import os
import time
from PIL import Image

class FaceSystem:
    def __init__(self, dataset_path="dataset", model_file="face_model.npz"):
        self.dataset_path = dataset_path
        self.model_file = model_file
        self.target_size = (100, 100)  # Kích thước chuẩn cho Eigenfaces
        self.num_components = 40       # Số lượng đặc trưng (Eigenvectors) giữ lại
        
        # Dữ liệu mô hình
        self.mean_face = None
        self.eigenfaces = None
        self.projections = []
        self.labels = []
        
        # Bộ phát hiện khuôn mặt (Haar Cascade - Nhanh và Nhẹ)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def preprocess(self, img_arr, is_training=False):
        """
        Quy trình chuẩn hóa ảnh:
        Gray -> (Auto Crop nếu Train) -> EqualizeHist -> Resize
        """
        # 1. Chuyển ảnh xám
        if len(img_arr.shape) > 2:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_arr

        # 2. Logic Cắt mặt (Chỉ áp dụng khi học dữ liệu thô)
        if is_training:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if len(faces) > 0:
                # Lấy khuôn mặt to nhất trong ảnh
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                gray = gray[y:y+h, x:x+w]
        
        # 3. Cân bằng sáng (Quan trọng để chống nhiễu sáng)
        gray = cv2.equalizeHist(gray)
        
        # 4. Resize về kích thước cố định
        gray = cv2.resize(gray, self.target_size)
        return gray

    def load_or_train(self):
        """Kiểm tra xem có model chưa. Có thì Load, chưa thì Train."""
        if os.path.exists(self.model_file):
            print(f"[INFO] Đã tìm thấy file model: {self.model_file}")
            try:
                data = np.load(self.model_file, allow_pickle=True)
                self.mean_face = data['mean_face']
                self.eigenfaces = data['eigenfaces']
                self.projections = data['projections']
                self.labels = data['labels']
                print("[OK] Load model thành công!")
                return True
            except:
                print("[WARN] File model lỗi, sẽ train lại...")
        
        return self.train()

    def train(self):
        print(f"\n[INFO] Đang quét dữ liệu từ: {self.dataset_path}")
        images_flat = []
        labels_list = []
        count = 0
        
        # Duyệt qua các thư mục con
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                # Chỉ nhận JPG, PNG, BMP (Bỏ HEIC cho nhẹ)
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    try:
                        path = os.path.join(root, file)
                        person_name = os.path.basename(root)
                        
                        # Đọc ảnh
                        pil_img = Image.open(path)
                        img_np = np.asarray(pil_img)
                        
                        # Xử lý
                        processed = self.preprocess(img_np, is_training=True)
                        
                        images_flat.append(processed.flatten())
                        labels_list.append(person_name)
                        count += 1
                        
                        if count % 50 == 0: print(f"  -> Đã học {count} ảnh...")
                    except:
                        continue

        if count == 0:
            print("[ERROR] Không tìm thấy ảnh nào! Hãy kiểm tra thư mục dataset.")
            return False

        print(f"[INFO] Đang tính toán Eigenfaces trên {count} ảnh...")
        
        # --- THUẬT TOÁN EIGENFACES (PCA) ---
        X = np.array(images_flat, dtype=np.float32)
        
        # 1. Mean Face
        self.mean_face = np.mean(X, axis=0)
        A = X - self.mean_face
        
        # 2. Covariance Matrix (Rút gọn)
        L = np.matmul(A, A.T)
        
        # 3. Eigen Decomposition
        eig_vals, eig_vecs_L = np.linalg.eigh(L)
        
        # Sắp xếp giảm dần
        idx = np.argsort(eig_vals)[::-1]
        eig_vecs_L = eig_vecs_L[:, idx]
        
        # Lấy K thành phần
        k = min(self.num_components, len(images_flat))
        k_vecs = eig_vecs_L[:, :k]
        
        # 4. Eigenfaces thực
        self.eigenfaces = np.matmul(A.T, k_vecs)
        for i in range(k): # Chuẩn hóa
            self.eigenfaces[:, i] /= np.linalg.norm(self.eigenfaces[:, i])
        self.eigenfaces = self.eigenfaces.T

        # 5. Chiếu dữ liệu (Projection)
        self.projections = np.matmul(A, self.eigenfaces.T)
        self.labels = labels_list
        
        # Lưu lại để lần sau không phải train nữa
        np.savez(self.model_file, 
                 mean_face=self.mean_face, 
                 eigenfaces=self.eigenfaces, 
                 projections=self.projections, 
                 labels=self.labels)
        
        print("[SUCCESS] Huấn luyện xong và đã lưu file!")
        return True

    def run(self, threshold=4000):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Không mở được Webcam.")
            return

        # Tối ưu FPS: Thu nhỏ ảnh 2 lần để tìm mặt nhanh hơn gấp 4 lần
        scale = 0.5
        
        print("\n" + "="*30)
        print(" CAMERA SẴN SÀNG HOẠT ĐỘNG")
        print(" [q]: Thoát")
        print(" [r]: Học lại dữ liệu mới")
        print(" [u]: Tăng ngưỡng (Dễ nhận hơn)")
        print(" [d]: Giảm ngưỡng (Khó nhận hơn)")
        print("="*30)

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1) # Lật như gương
            
            # 1. Resize ảnh nhỏ để Detect nhanh
            small_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # 2. Detect khuôn mặt
            faces = self.face_cascade.detectMultiScale(gray_small, 1.1, 5, minSize=(30,30))

            for (xs, ys, ws, hs) in faces:
                # Quy đổi tọa độ về ảnh gốc
                x, y, w, h = int(xs/scale), int(ys/scale), int(ws/scale), int(hs/scale)
                
                # Vẽ khung
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 3. Nhận dạng (Cắt từ ảnh gốc sắc nét)
                face_roi = frame[y:y+h, x:x+w]
                
                try:
                    # Xử lý
                    processed = self.preprocess(face_roi, is_training=False)
                    
                    # Tính toán khoảng cách
                    img_flat = processed.flatten().astype(np.float32)
                    img_norm = img_flat - self.mean_face
                    proj_new = np.matmul(self.eigenfaces, img_norm)
                    
                    # So khớp Vector (Vectorization - Siêu nhanh)
                    dists = np.linalg.norm(self.projections - proj_new, axis=1)
                    min_index = np.argmin(dists)
                    min_dist = dists[min_index]
                    
                    # Kiểm tra ngưỡng
                    if min_dist < threshold:
                        name = self.labels[min_index]
                        color = (0, 255, 0)
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)
                    
                    # Hiển thị
                    text = f"{name} ({int(min_dist)})"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                except:
                    pass # Bỏ qua nếu lỗi tính toán nhỏ

            # Hiển thị thông số
            cv2.putText(frame, f"Threshold: {threshold}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Face Recognition (Lightweight)", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'): self.train() # Train nóng
            elif key == ord('u'): threshold += 200
            elif key == ord('d'): threshold -= 200

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # ĐƯỜNG DẪN DATASET CỦA BẠN
    DATA_DIR = "dataset"
    
    app = FaceSystem(dataset_path=DATA_DIR)
    
    # Bước 1: Load hoặc Train
    if app.load_or_train():
        # Bước 2: Chạy Camera (Ngưỡng mặc định 4000)
        app.run(threshold=6000)