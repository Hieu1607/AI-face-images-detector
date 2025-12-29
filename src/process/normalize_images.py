import numpy as np

def standardize_normalize(image_array: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa giá trị pixel từ [0, 255] sang [0.0, 1.0].

    Args:
        image_array (np.ndarray): Ảnh đầu vào (thường là uint8).

    Returns:
        np.ndarray: Ảnh đã được chuẩn hóa (float32).
    """
    # Chuyển đổi kiểu dữ liệu sang float32 để thực hiện phép chia
    image_float = image_array.astype(np.float32)
    
    # Chia cho 255.0 để chuẩn hóa
    normalized_image = image_float / 255.0
    
    return normalized_image

# Ví dụ:
# image_normalized = standardize_normalize(image_224)