import cv2
import threading
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from norfair import Detection, Tracker

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
DB_PATH = "dataset"             # Thư mục chứa dữ liệu ảnh
MODEL_NAME = "VGG-Face"    # Model nhận diện (Cân bằng tốt nhất)
DISTANCE_THRESHOLD = 30    # Khoảng cách pixel để Tracker nhận diện là 1 người
CONFIDENCE_STOP = 0.95     # Nếu độ tin cậy > 95% thì không cần nhận diện lại quá nhiều

# ==========================================
# 1. KHỞI TẠO CÁC MODEL AI
# ==========================================

# A. MediaPipe (Dùng để phát hiện khuôn mặt siêu tốc)
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# B. Norfair (Dùng để theo dõi đối tượng - Tracking)
def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

tracker = Tracker(distance_function=euclidean_distance, 
                  distance_threshold=DISTANCE_THRESHOLD)

# C. Bộ nhớ chia sẻ (Lưu trạng thái tên của các ID)
# Format: { track_id (int) : "Tên người" (string) }
id_name_mapping = {} 
processing_ids = [] # Danh sách các ID đang được AI xử lý

# ==========================================
# 2. HÀM XỬ LÝ NHẬN DIỆN (CHẠY NGẦM)
# ==========================================
def recognize_worker(face_img, track_id):
    """
    Hàm này chạy trong luồng riêng (Thread) để không làm đơ Camera
    """
    global id_name_mapping, processing_ids
    
    try:
        # Gọi DeepFace để tìm kiếm trong DB
        results = DeepFace.find(img_path=face_img, 
                                db_path=DB_PATH, 
                                model_name=MODEL_NAME, 
                                detector_backend='mediapipe',
                                enforce_detection=False, 
                                silent=True)
        
        found_name = "Unknown"
        if len(results) > 0 and not results[0].empty:
            # Lấy đường dẫn ảnh khớp nhất
            path = results[0].iloc[0]['identity']
            # Trích xuất tên thư mục (chính là tên người)
            import os
            found_name = os.path.basename(os.path.dirname(path))
        
        # Cập nhật vào bộ nhớ
        id_name_mapping[track_id] = found_name
        
    except Exception as e:
        print(f"Error ID {track_id}: {e}")
    finally:
        # Xóa ID khỏi danh sách đang bận
        if track_id in processing_ids:
            processing_ids.remove(track_id)

# ==========================================
# 3. VÒNG LẶP CHÍNH (MAIN LOOP)
# ==========================================
def main():
    cap = cv2.VideoCapture(0) # Mở Webcam
    
    print("Hệ thống đang khởi động... Vui lòng chờ giây lát!")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Lật ảnh cho giống gương
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # --- BƯỚC 1: PHÁT HIỆN MẶT (MediaPipe) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb_frame)
        
        norfair_detections = []
        
        if results.detections:
            for det in results.detections:
                bboxC = det.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Tính tâm (centroid) để Tracking
                center_x = x + width // 2
                center_y = y + height // 2
                
                # Tạo đối tượng Detection cho Norfair
                norfair_detections.append(Detection(points=np.array([center_x, center_y]), 
                                                    data=[x, y, width, height]))

        # --- BƯỚC 2: TRACKING (Norfair) ---
        tracked_objects = tracker.update(detections=norfair_detections)
        
        # --- BƯỚC 3: LOGIC VẼ VÀ GỌI AI ---
        for obj in tracked_objects:
            track_id = obj.id
            
            if obj.last_detection is None: continue
            
            # Lấy toạ độ box từ dữ liệu đã lưu
            x, y, width, height = obj.last_detection.data
            
            # Kiểm tra xem ID này đã biết tên chưa
            name = id_name_mapping.get(track_id, "Scanning...")
            
            # Nếu chưa biết tên VÀ đang không bận xử lý -> Gọi AI
            if (track_id not in id_name_mapping) and (track_id not in processing_ids):
                processing_ids.append(track_id)
                
                # Cắt ảnh khuôn mặt (thêm padding tí cho dễ nhìn)
                pad = 10
                face_crop = frame[max(0, y-pad):min(h, y+height+pad), 
                                  max(0, x-pad):min(w, x+width+pad)]
                
                if face_crop.size > 0:
                    threading.Thread(target=recognize_worker, args=(face_crop, track_id)).start()

            # Chọn màu sắc
            color = (0, 255, 0) # Xanh (Đã nhận diện)
            if name == "Unknown": color = (0, 0, 255) # Đỏ (Người lạ)
            if name == "Scanning...": color = (0, 255, 255) # Vàng (Đang đọc)

            # Vẽ khung và tên
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.putText(frame, f"ID:{track_id} {name}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Hiển thị thông tin
        cv2.putText(frame, "Nhan 'Q' de thoat", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Face Recognition System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()