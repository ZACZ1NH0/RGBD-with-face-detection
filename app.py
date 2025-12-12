import base64
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request, jsonify
import os
# ==========================================
# 1. SETUP MODEL (FaceModel + MiDaS Local)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("⏳ Đang tải MiDaS từ Local...")

# 1. Load kiến trúc model từ folder source code (source='local')
# Lưu ý: 'midas_src' là tên folder bạn vừa clone về
midas = torch.hub.load(repo_or_dir='midas_src', model='MiDaS_small', source='local', pretrained=False)

# 2. Load weights từ file .pt đã tải
midas_weight_path = "midas_small.pt" # Đường dẫn file weights
midas.load_state_dict(torch.load(midas_weight_path, map_location=DEVICE))

midas.to(DEVICE)
midas.eval()

# 3. Load Transforms (Bộ tiền xử lý ảnh) từ local luôn
midas_transforms = torch.hub.load(repo_or_dir='midas_src', model='transforms', source='local').small_transform

print("✅ MiDaS (Local) đã sẵn sàng!")

# --- B. Class FaceModelInference (Của bạn) ---
class FaceModelInference(nn.Module):
    def __init__(self):
        super().__init__()
        # RGB
        base_rgb = models.resnet18(weights=None)
        self.rgb_backbone = nn.Sequential(*list(base_rgb.children())[:-1])
        self.rgb_projector = nn.Linear(512, 512)
        # Depth
        base_depth = models.efficientnet_b0(weights=None)
        base_depth.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.depth_features = base_depth.features
        self.depth_pool = nn.AdaptiveAvgPool2d(1)
        self.depth_projector = nn.Linear(1280, 512)
        # Fusion
        self.fusion_head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512)
        )

    def forward(self, rgb, depth):
        x_rgb = self.rgb_projector(self.rgb_backbone(rgb).view(rgb.size(0), -1))
        x_d = self.depth_projector(self.depth_pool(self.depth_features(depth)).flatten(1))
        x_final = self.fusion_head(torch.cat([F.normalize(x_rgb), F.normalize(x_d)], dim=1))
        return F.normalize(x_final)

# --- C. Khởi tạo Hệ thống ---
print("⏳ Đang tải Face Model...")
face_model = FaceModelInference().to(DEVICE)
# Load weights của bạn (Đảm bảo file .pth nằm cùng thư mục)
if torch.cuda.is_available():
    face_model.load_state_dict(torch.load("fusion_face_final.pth"))
else:
    face_model.load_state_dict(torch.load("fusion_face_final.pth", map_location="cpu"))
face_model.eval()

# ==========================================
# CẤU HÌNH LƯU TRỮ DỮ LIỆU
# ==========================================
DB_FILE = "user_db.pt"  # Tên file chứa dữ liệu người dùng

def load_db():
    """Hàm load dữ liệu từ ổ cứng khi khởi động"""
    if os.path.exists(DB_FILE):
        print(f"📂 Tìm thấy file dữ liệu: {DB_FILE}. Đang tải...")
        try:
            # torch.load cực tiện, lưu được cả Tensor lẫn Dictionary
            return torch.load(DB_FILE)
        except Exception as e:
            print(f"⚠️ File lỗi, tạo DB mới. Lỗi: {e}")
            return {}
    else:
        print("✨ Chưa có dữ liệu cũ. Tạo database mới...")
        return {}

def save_db():
    """Hàm lưu dữ liệu xuống ổ cứng"""
    try:
        torch.save(face_database, DB_FILE)
        print("💾 Đã lưu dữ liệu người dùng thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi lưu dữ liệu: {e}")

# Database giả lập (Lưu trên RAM, tắt server là mất -> Thực tế nên lưu file/DB)
face_database = load_db()

# Transforms cho Face Model
tf_rgb = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
tf_depth = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ==========================================
# 2. FLASK APP
# ==========================================
app = Flask(__name__)

def generate_depth(rgb_img_pil):
    """Hàm phù thủy: Biến ảnh RGB thành ảnh Depth bằng AI"""
    img_cv = np.array(rgb_img_pil) 
    input_batch = midas_transforms(img_cv).to(DEVICE)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(224, 224), # Resize về đúng kích thước model Face cần
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Chuẩn hóa về ảnh Grayscale (0-255) để giống ảnh Depth thật
    depth_numpy = prediction.cpu().numpy()
    depth_min = depth_numpy.min()
    depth_max = depth_numpy.max()
    depth_normalized = (depth_numpy - depth_min) / (depth_max - depth_min)
    depth_pil = Image.fromarray((depth_normalized * 255).astype("uint8"), mode="L")
    
    return depth_pil

def get_embedding(rgb_pil):
    # 1. Tạo Depth từ RGB
    depth_pil = generate_depth(rgb_pil)
    
    # 2. Transform
    rgb_t = tf_rgb(rgb_pil).unsqueeze(0).to(DEVICE)
    depth_t = tf_depth(depth_pil).unsqueeze(0).to(DEVICE)
    
    # 3. Forward
    with torch.no_grad():
        emb = face_model(rgb_t, depth_t)
    
    # Trả về cả ảnh depth để hiển thị cho ngầu
    return emb, depth_pil

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    image_data = data['image'].split(",")[1]
    name = data['name']
    
    # Decode ảnh
    img_bytes = base64.b64decode(image_data)
    rgb_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    
    # Lấy embedding
    emb, depth_pil = get_embedding(rgb_pil)
    
    # Lưu vào DB
    if name not in face_database:
        face_database[name] = []
    face_database[name].append(emb.cpu())
    
    save_db()

    return jsonify({"message": f"✅ Đã đăng ký: {name}", "status": "success"})

@app.route('/identify', methods=['POST'])
def identify():
    data = request.json
    image_data = data['image'].split(",")[1]
    
    # Decode
    img_bytes = base64.b64decode(image_data)
    rgb_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    
    # Lấy embedding của ảnh input
    unknown_emb, depth_pil = get_embedding(rgb_pil)
    unknown_emb_cpu = unknown_emb.cpu()
    best_name = "Người lạ"
    min_dist = 100.0
    
    # So sánh
    for name, vectors in face_database.items():
        for vec in vectors:
            dist = (unknown_emb_cpu - vec).pow(2).sum().sqrt().item()
            if dist < min_dist:
                min_dist = dist
                best_name = name
    
    # Convert ảnh depth sang base64 để trả về client hiển thị
    buffered = io.BytesIO()
    depth_pil.save(buffered, format="JPEG")
    depth_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    threshold = 0.8
    msg = f"✅ Xin chào: {best_name}" if min_dist < threshold else "❌ Không nhận ra"
    
    return jsonify({
        "message": f"{msg} (Dist: {min_dist:.2f})",
        "depth_url": f"data:image/jpeg;base64,{depth_b64}",
        "rgb_url": data['image']
    })

if __name__ == '__main__':
    # Chạy server
    app.run(host='0.0.0.0', port=5000, debug=True)