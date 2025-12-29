import cv2
import numpy as np

def load_image_with_opencv(image_path: str) -> np.ndarray:
    """
    Tải ảnh từ đường dẫn và chuyển thành NumPy Array (H x W x C).
    
    Lưu ý: OpenCV mặc định tải ảnh ở định dạng BGR (Blue-Green-Red).
    """
    # cv2.IMREAD_COLOR đảm bảo ảnh được tải ở dạng màu (nếu có)
    image_array_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR) 
    
    if image_array_bgr is None:
        raise FileNotFoundError(f"Không tìm thấy hoặc không thể đọc file ảnh tại: {image_path}")
    
    # Chuyển đổi từ BGR sang RGB
    image_rgb = cv2.cvtColor(image_array_bgr, cv2.COLOR_BGR2RGB)        
    return image_rgb

def resize_image(image_array: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Thay đổi kích thước ảnh.

    Args:
        image_array (np.ndarray): Ảnh đầu vào (ví dụ: kích thước H x W x C).
        target_size (tuple): Kích thước mục tiêu (W, H), ví dụ: (224, 224).

    Returns:
        np.ndarray: Ảnh đã được thay đổi kích thước.
    """
    # cv2.resize nhận kích thước theo định dạng (width, height)
    resized_image = cv2.resize(image_array, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_image

# Ví dụ:
# image_224 = resize_image(my_original_image, (224, 224))

def random_crop(image_array: np.ndarray, crop_width : float = 0.5, crop_height: float = 0.5) -> np.ndarray:
    """
    Cắt ngẫu nhiên một vùng từ ảnh.

    Args:
        image_array (np.ndarray): Ảnh đầu vào (H x W x C).
        crop_size (tuple): Kích thước vùng cắt (width, height).

    Returns:
        np.ndarray: Vùng ảnh đã được cắt.
    """
    img_height, img_width = image_array.shape[:2]

    if crop_width < 0 or crop_width > 1.0:
        raise ValueError("crop_width phải nằm trong khoảng (0, 1.0]")
    if crop_height < 0 or crop_height > 1.0:
        raise ValueError("crop_height phải nằm trong khoảng (0, 1.0]")

    # Tính kích thước cắt (chuyển về integer)
    crop_w = int(crop_width * img_width)
    crop_h = int(crop_height * img_height)

    # Chọn tọa độ ngẫu nhiên để cắt
    x_start = np.random.randint(0, img_width - crop_w + 1)
    y_start = np.random.randint(0, img_height - crop_h + 1)

    # Cắt ảnh
    cropped_image = image_array[y_start:y_start + crop_h, x_start:x_start + crop_w]
    return cropped_image