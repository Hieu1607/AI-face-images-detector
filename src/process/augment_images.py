import cv2
import numpy as np
import random
def augment_flip(image_array: np.ndarray, flip_code: int = 1) -> np.ndarray:
    """
    Thực hiện lật ảnh.
    
    Args:
        image_array: np.ndarray ảnh đầu vào.
        flip_code: 1 (Lật ngang), 0 (Lật dọc), -1 (Lật cả hai).
        
    Returns:
        np.ndarray: Ảnh đã lật.
    """
    # cv2.flip là hàm thường dùng và nhanh chóng
    flipped_image = cv2.flip(image_array, flip_code)
    return flipped_image

# Ví dụ sử dụng:
# image_flipped_horizontal = augment_flip(my_image, flip_code=1) 
# image_flipped_vertical = augment_flip(my_image, flip_code=0)

def augment_rotation(image_array: np.ndarray, angle: float) -> np.ndarray:
    """
    Xoay ảnh quanh tâm với một góc (độ) cho trước.
    
    Args:
        image_array: np.ndarray ảnh đầu vào.
        angle: Góc xoay (ví dụ: 15, -10).
        
    Returns:
        np.ndarray: Ảnh đã xoay.
    """
    (h, w) = image_array.shape[:2]
    center = (w // 2, h // 2)
    
    # Lấy ma trận xoay (Rotation Matrix)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Áp dụng phép xoay
    rotated_image = cv2.warpAffine(image_array, M, (w, h))
    return rotated_image

# Ví dụ sử dụng:
# image_rotated = augment_rotation(my_image, angle=random.uniform(-10, 10))

def augment_shift(image_array: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """
    Dịch chuyển ảnh theo một khoảng cố định.
    
    Args:
        image_array: np.ndarray ảnh đầu vào.
        shift_x: Số pixel dịch chuyển ngang (âm: trái, dương: phải).
        shift_y: Số pixel dịch chuyển dọc (âm: lên, dương: xuống).
        
    Returns:
        np.ndarray: Ảnh đã dịch chuyển.
    """
    (h, w) = image_array.shape[:2]
    
    # Ma trận dịch chuyển 2x3 (Translation Matrix)
    M = np.float32([[1, 0, shift_x], 
                    [0, 1, shift_y]])
    
    # Áp dụng phép biến đổi affine
    shifted_image = cv2.warpAffine(image_array, M, (w, h))
    return shifted_image

# Ví dụ sử dụng:
# shift_amount = 10 
# image_shifted = augment_shift(my_image, shift_x=random.randint(-shift_amount, shift_amount), 
#                                         shift_y=random.randint(-shift_amount, shift_amount))

def augment_brightness(image_array: np.ndarray, factor: int) -> np.ndarray:
    """
    Điều chỉnh độ sáng bằng cách cộng/trừ một giá trị.
    
    Args:
        image_array: np.ndarray ảnh đầu vào (uint8).
        factor: Giá trị cộng thêm (ví dụ: 30 để sáng hơn, -30 để tối hơn).
        
    Returns:
        np.ndarray: Ảnh đã điều chỉnh độ sáng.
    """
    # Dùng int16 thay vì int64 để tránh tràn số + tiết kiệm RAM
    bright_image = image_array.astype(np.int16)
    bright_image += factor

    # Clip trực tiếp trên buffer hiện tại để tránh tạo bản sao mới
    np.clip(bright_image, 0, 255, out=bright_image)

    return bright_image.astype(np.uint8)


# Ví dụ sử dụng:
# brightness_factor = random.randint(-40, 40)
# image_brightened = augment_brightness(my_image, factor=brightness_factor)

def augment_contrast(image_array: np.ndarray, factor: float) -> np.ndarray:
    """
    Điều chỉnh độ tương phản bằng cách nhân các giá trị pixel.
    
    Args:
        image_array: np.ndarray ảnh đầu vào (thường là uint8).
        factor: Hệ số tương phản (ví dụ: 1.5 để tăng, 0.5 để giảm).
        
    Returns:
        np.ndarray: Ảnh đã điều chỉnh độ tương phản.
    """
    # Nhân với hệ số (phải chuyển sang float) và clip lại giá trị
    contrast_image = np.clip(image_array.astype(float) * factor, 0, 255).astype(np.uint8)
    return contrast_image

# Ví dụ sử dụng:
# contrast_factor = random.uniform(0.7, 1.3) # Thay đổi từ 70% đến 130%
# image_contrasted = augment_contrast(my_image, factor=contrast_factor)

def augment_add_noise(image_array: np.ndarray, std_dev: float = 20.0) -> np.ndarray:
    """
    Thêm nhiễu Gaussian (nhiễu ngẫu nhiên) vào ảnh.
    
    Args:
        image_array: np.ndarray ảnh đầu vào (thường là uint8).
        std_dev: Độ lệch chuẩn của nhiễu Gaussian.
        
    Returns:
        np.ndarray: Ảnh đã có nhiễu.
    """
    # Chuyển ảnh sang float để tính toán nhiễu
    image_float = image_array.astype(np.float32)
    
    # Tạo nhiễu Gaussian có cùng kích thước với ảnh
    mean = 0.0
    noise = np.random.normal(mean, std_dev, image_float.shape).astype(np.float32)
    
    # Cộng nhiễu và clip lại giá trị
    noisy_image = image_float + noise
    
    # Đảm bảo giá trị pixel nằm trong [0, 255] và chuyển về uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

# Ví dụ sử dụng:
# image_noisy = augment_add_noise(my_image, std_dev=random.uniform(5, 25))

# (Giả định: Các hàm augment_flip, augment_rotation, augment_shift, 
# augment_brightness, augment_contrast, augment_add_noise đã được định nghĩa ở trên)

def random_augmentation_pipeline(image_array: np.ndarray) -> np.ndarray:
    """
    Tạo một pipeline để áp dụng ngẫu nhiên các kỹ thuật tăng cường dữ liệu ảnh.

    Args:
        image_array (np.ndarray): Ảnh đầu vào (H x W x C, uint8).

    Returns:
        np.ndarray: Ảnh đã được tăng cường ngẫu nhiên.
    """
    # Tạo bản sao để tránh thay đổi ảnh gốc
    augmented_image = image_array.copy()

    # --- 1. Tăng cường Hình học (Geometric Augmentation) ---

    # Lật Ngang (Horizontal Flip)
    if random.random() < 0.5: # 50% cơ hội lật ngang
        augmented_image = augment_flip(augmented_image, flip_code=1)
        
    # Xoay (Rotation)
    if random.random() < 0.5: # 50% cơ hội xoay
        angle = random.uniform(-15, 15)
        augmented_image = augment_rotation(augmented_image, angle=angle)
        
    # Dịch chuyển (Shifting)
    if random.random() < 0.5: # 50% cơ hội dịch chuyển
        max_shift = 15 # Dịch chuyển tối đa 15 pixel
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        augmented_image = augment_shift(augmented_image, shift_x, shift_y)

    # --- 2. Tăng cường Màu sắc và Độ sáng (Color Augmentation) ---
    
    # Điều chỉnh Độ sáng (Brightness)
    if random.random() < 0.5: # 50% cơ hội thay đổi độ sáng
        brightness_factor = random.randint(-30, 30)
        augmented_image = augment_brightness(augmented_image, factor=brightness_factor)

    # Điều chỉnh Độ tương phản (Contrast)
    if random.random() < 0.5: # 50% cơ hội thay đổi độ tương phản
        contrast_factor = random.uniform(0.7, 1.3) # Từ 70% đến 130%
        augmented_image = augment_contrast(augmented_image, factor=contrast_factor)
        
    # Thêm Nhiễu (Noise)
    if random.random() < 0.5: # 50% cơ hội thêm nhiễu
        std_dev = random.uniform(5, 20)
        augmented_image = augment_add_noise(augmented_image, std_dev=std_dev)

    # Ghi chú: Kỹ thuật Random Cropping thường được xử lý riêng 
    # trong quá trình tải dữ liệu, thường là bước cuối cùng sau khi Resize.

    return augmented_image