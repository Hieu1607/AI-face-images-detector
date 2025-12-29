import zipfile
import os
import gdown
# --- Thiết lập Cấu hình ---
# Thay thế CHUỖI_ID_FILE_CỦA_BẠN bằng ID thực của file Zip trên Google Drive
# ID là phần nằm giữa "/d/" và "/view" trong link chia sẻ.
FILE_ID = 'CHUỖI_ID_FILE_CỦA_BẠN'
DESTINATION_PATH = './downloaded_file.zip'
EXTRACT_FOLDER = './demo_data'

# --- 1. Hàm Tải File từ Google Drive ---
def download_file_from_google_drive(file_id, destination):
    """
    Tải file từ Google Drive, vượt qua bước kiểm tra virus cho file lớn.
    """
    print(f"Bắt đầu tải file ID: {file_id}...")
    
    # Tạo thư mục đích nếu nó chưa tồn tại
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # URL cơ bản để tải file trực tiếp với confirm parameter
    URL = f"https://drive.google.com/uc?id={file_id}"

    result = gdown.download(URL, destination, quiet=False)

    if not result or not os.path.exists(destination):
        print("Lỗi: Không thể tải file từ Google Drive.")

    # Kiểm tra kích thước file
    file_size = os.path.getsize(destination)
    filename = os.path.basename(destination)
    print(f"Đã tải xong: {filename}")
    print(f"Kích thước: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    print()

# --- Hàm lấy token xác nhận cho file lớn ---
def get_confirm_token(response):
    # First try to get from cookies
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    
    # If not found in cookies, try to parse from HTML content
    try:
        content = response.text
        if 'confirm=' in content:
            import re
            match = re.search(r'confirm=([^&"\']+)', content)
            if match:
                return match.group(1)
    except:
        pass
    
    return None

# --- 2. Hàm Giải Nén File Zip ---
def unzip_file(zip_path, extract_to):
    """
    Giải nén file zip vào thư mục chỉ định.
    """
    print(f"Bắt đầu giải nén file: {zip_path}")
    
    # Kiểm tra file có tồn tại không
    if not os.path.exists(zip_path):
        print(f"Lỗi: File không tồn tại: {zip_path}")
        return False
    
    # Kiểm tra kích thước file
    file_size = os.path.getsize(zip_path)
    print(f"Kích thước file: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    # Kiểm tra nếu file quá nhỏ (có thể là HTML error page)
    if file_size < 1024:  # Nhỏ hơn 1KB
        print("CẢNH BÁO: File có kích thước quá nhỏ, có thể không phải zip file thật.")
        # Đọc nội dung để kiểm tra
        with open(zip_path, 'rb') as f:
            content = f.read(500)
            try:
                content_str = content.decode('utf-8', errors='ignore')
                if '<html>' in content_str.lower() or '<!doctype html>' in content_str.lower():
                    print("Lỗi: File tải về là trang HTML, không phải zip file!")
                    print("Nội dung file:", content_str[:200])
                    return False
            except:
                pass
    
    # Kiểm tra file header để xác định có phải zip file không
    try:
        with open(zip_path, 'rb') as f:
            header = f.read(4)
            # Zip file magic numbers: PK (0x504B)
            if not (header[:2] == b'PK' or header[:4] == b'\x50\x4b\x03\x04'):
                print("Lỗi: File không có header của zip file!")
                print(f"Header hiện tại: {header}")
                # Thử đọc content để debug
                f.seek(0)
                content = f.read(200).decode('utf-8', errors='ignore')
                print("Nội dung file:", content[:100])
                return False
    except Exception as e:
        print(f"Lỗi khi đọc file header: {e}")
        return False
    
    # Tạo thư mục đích nếu nó chưa tồn tại
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        # Kiểm tra zipfile có hợp lệ không trước khi giải nén
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Test zip file integrity
            zip_ref.testzip()
            # Nếu không có lỗi, tiến hành giải nén
            zip_ref.extractall(extract_to)
        print(f"Giải nén thành công vào thư mục: {extract_to}")
        return True
    except zipfile.BadZipFile:
        print("Lỗi: File tải về không phải là file Zip hợp lệ hoặc bị hỏng.")
        return False
    except Exception as e:
        print(f"Đã xảy ra lỗi khi giải nén: {e}")
        return False

# --- 3. Thực thi Chương trình Chính ---
if __name__ == "__main__":
    # --- BƯỚC 1: Tải file ---
    # !!! Đảm bảo bạn đã thay thế FILE_ID ở trên !!!
    FILE_ID = '1xd4k7GLmR6kIC6RRf84AtqZk7hVtH41T'
    if FILE_ID == 'CHUỖI_ID_FILE_CỦA_BẠN':
        print("LỖI: Vui lòng thay thế 'CHUỖI_ID_FILE_CỦA_BẠN' bằng ID Google Drive thực tế.")
    else:
        download_file_from_google_drive(FILE_ID, DESTINATION_PATH)
        
        # --- BƯỚC 2: Giải nén file ---
        if os.path.exists(DESTINATION_PATH):
            unzip_file(DESTINATION_PATH, EXTRACT_FOLDER)
            
            # (Tùy chọn) Xóa file zip sau khi giải nén
            # os.remove(DESTINATION_PATH)
            # print(f"Đã xóa file zip: {DESTINATION_PATH}")
        else:
            print("Không thể giải nén vì file tải xuống không tồn tại.")