import cv2
import numpy as np
import os

def augment_data(dataset_path):
    print("--- ĐANG TỰ ĐỘNG TẠO THÊM DỮ LIỆU ---")
    
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir): continue
        
        # Lấy danh sách ảnh hiện có
        files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f" -> Xử lý '{person_name}': {len(files)} ảnh gốc")
        
        count_new = 0
        for fname in files:
            img_path = os.path.join(person_dir, fname)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # 1. TẠO ẢNH LẬT NGANG (Flip)
            # Giúp máy học được khuôn mặt đối xứng
            img_flip = cv2.flip(img, 1)
            new_name = f"aug_flip_{fname}"
            cv2.imwrite(os.path.join(person_dir, new_name), img_flip)
            count_new += 1
            
            # 2. TẠO ẢNH SÁNG HƠN (Brightness +)
            # Giúp nhận dạng tốt khi bị chói
            M = np.ones(img.shape, dtype="uint8") * 40
            img_bright = cv2.add(img, M)
            new_name = f"aug_bright_{fname}"
            cv2.imwrite(os.path.join(person_dir, new_name), img_bright)
            count_new += 1

            # 3. TẠO ẢNH TỐI HƠN (Brightness -)
            # Giúp nhận dạng tốt trong bóng tối
            M = np.ones(img.shape, dtype="uint8") * 40
            img_dark = cv2.subtract(img, M)
            new_name = f"aug_dark_{fname}"
            cv2.imwrite(os.path.join(person_dir, new_name), img_dark)
            count_new += 1
            
            # 4. TẠO ẢNH NHIỄU (Noise)
            # Giả lập camera chất lượng thấp (webcam mờ)
            row, col, ch = img.shape
            mean = 0
            var = 100
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = img + gauss
            # Clip về đoạn [0, 255]
            noisy = np.clip(noisy, 0, 255).astype("uint8")
            
            new_name = f"aug_noise_{fname}"
            cv2.imwrite(os.path.join(person_dir, new_name), noisy)
            count_new += 1

        print(f"    -> Đã tạo thêm {count_new} ảnh mới.")

if __name__ == "__main__":
    DATA_PATH = "dataset" # Đường dẫn tới thư mục ảnh của bạn
    augment_data(DATA_PATH)