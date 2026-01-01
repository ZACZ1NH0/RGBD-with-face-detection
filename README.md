# 🎭 RGB-D Face Recognition System (Fusion Network)

> Hệ thống nhận diện khuôn mặt đa phương thức sử dụng **RGB** kết hợp **Depth (Độ sâu)**, ứng dụng **One-Shot Learning** với Triplet Loss.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)

## 🌟 Giới thiệu

Dự án này giải quyết bài toán nhận diện khuôn mặt bằng cách kết hợp hai luồng thông tin:
1.  **RGB (Màu sắc):** Trích xuất đặc trưng ngoại quan.
2.  **Depth (Độ sâu):** Trích xuất đặc trưng hình học 3D, giúp chống lại việc giả mạo bằng ảnh in (anti-spoofing).

Điểm đặc biệt: Hệ thống tích hợp **MiDaS (Monocular Depth Estimation)**, cho phép sử dụng Webcam thông thường để tạo ra ảnh Depth giả lập theo thời gian thực mà không cần camera 3D chuyên dụng.

## 🚀 Tính năng chính

* **Fusion Architecture:** Kết hợp ResNet18 (RGB) và EfficientNet-B0 (Depth).
* **One-Shot Learning:** Chỉ cần **1 bức ảnh mẫu** để đăng ký người dùng mới.
* **Open-Set Recognition:** Nhận diện được người lạ (Unknown) chưa từng xuất hiện trong tập huấn luyện.
* **Real-time Web App:** Giao diện Flask thân thiện, hỗ trợ chụp ảnh và nhận diện trực tiếp.
* **Persistent Database:** Tự động lưu trữ dữ liệu người dùng đã đăng ký vào ổ cứng.
* **Offline Ready:** Tích hợp mô hình MiDaS chạy local, không phụ thuộc internet.

## 🧠 Kiến trúc Mô hình



1.  **RGB Encoder:** ResNet18 (Pretrained ImageNet) -> Output 512 dim.
2.  **Depth Encoder:** EfficientNet-B0 (Modified 1-channel input) -> Output 512 dim.
3.  **Fusion Head:** Nối (Concat) 2 vector -> Linear Layer -> Output 512 dim (Final Embedding).
4.  **Loss Function:** Triplet Loss (Margin = 1.0).

## 📂 Cấu trúc Dự án

```text
flask_face_app/
├── app.py                  # Server Flask chính (Chạy cái này để dùng)
├── train.py                # Script huấn luyện mô hình (Triplet Loss)
├── requirements.txt        # Danh sách thư viện
├── fusion_face_final.pth   # Weights mô hình nhận diện (Sau khi train)
├── midas_small.pt          # Weights mô hình tạo Depth (Tải về)
├── user_db.pt              # Database người dùng (Tự sinh ra)
├── midas_src/              # Source code MiDaS (Clone từ Github)
├── templates/
│   └── index.html          # Giao diện Web
└── static/
```
## Cài đặt
### Tạo môi trường ảo với Python 3.11 (nếu chưa tạo)

> ⚠️ Đảm bảo bạn đã cài Python 3.11 trước đó.

```bash
py -3.11 -m venv venv
```

Kích hoạt môi trường ảo:

- **Windows (PowerShell):**

```powershell
.\venv\Scripts\activate.ps1
```

- **Windows (CMD):**

```cmd
.\venv\Scripts\activate.bat
```
hoặc
```cmd
.\venv\Scripts\activate
```

- **macOS/Linux:**

```bash
source venv/bin/activate
```

---

### Cài đặt các thư viện phụ thuộc

```bash
pip install -r requirements.txt
```
### Chạy
```bash
python app.py
```
