import base64
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
import os
import threading
from facenet_pytorch import MTCNN

# ==========================================
# 1. CẤU HÌNH & KHỞI TẠO MODEL
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Running on: {DEVICE}")

# --- A. MTCNN (Face Detector) ---
print("⏳ Đang tải MTCNN (Face Detector)...")
# margin=20: Lấy rộng ra một chút để không cắt mất cằm/trán
mtcnn = MTCNN(keep_all=False, device=DEVICE, margin=20)
print("✅ MTCNN đã sẵn sàng!")

# --- B. MiDaS (Depth Estimation) ---
print("⏳ Đang tải MiDaS Local...")
# Lưu ý: Cần folder 'midas_src' và file 'midas_small.pt' ở cùng thư mục
try:
    midas = torch.hub.load(repo_or_dir='midas_src', model='MiDaS_small', source='local', pretrained=False)
    midas.load_state_dict(torch.load("midas_small.pt", map_location=DEVICE))
    midas.to(DEVICE)
    midas.eval()
    midas_transforms = torch.hub.load(repo_or_dir='midas_src', model='transforms', source='local').small_transform
    print("✅ MiDaS đã sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi tải MiDaS: {e}")
    exit()

# --- C. Face Model (Fusion Architecture) ---
class FaceModelInference(nn.Module):
    def __init__(self):
        super().__init__()
        # RGB Backbone
        base_rgb = models.resnet18(weights=None)
        self.rgb_backbone = nn.Sequential(*list(base_rgb.children())[:-1])
        self.rgb_projector = nn.Linear(512, 512)
        
        # Depth Backbone
        base_depth = models.efficientnet_b0(weights=None)
        base_depth.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.depth_features = base_depth.features
        self.depth_pool = nn.AdaptiveAvgPool2d(1)
        self.depth_projector = nn.Linear(1280, 512)
        
        # Fusion Head
        self.fusion_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, rgb, depth):
        # RGB Stream
        x_rgb = self.rgb_backbone(rgb).view(rgb.size(0), -1)
        x_rgb = self.rgb_projector(x_rgb)
        x_rgb = F.normalize(x_rgb)

        # Depth Stream
        x_d = self.depth_features(depth)
        x_d = self.depth_pool(x_d).flatten(1)
        x_d = self.depth_projector(x_d)
        x_d = F.normalize(x_d)

        # Fusion
        concat = torch.cat([x_rgb, x_d], dim=1)
        x_final = self.fusion_head(concat)
        return F.normalize(x_final)

print("⏳ Đang tải Face Model (Fusion)...")
face_model = FaceModelInference().to(DEVICE)

# --- LOAD WEIGHTS CỦA BẠN ---
# Hãy chắc chắn file .pth nằm cùng thư mục
MODEL_PATH = "fusion_face_finalv3.pth" # Hoặc "model.pth" tùy tên file của bạn
if os.path.exists(MODEL_PATH):
    face_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("✅ Face Model weights loaded!")
else:
    print(f"❌ CẢNH BÁO: Không tìm thấy {MODEL_PATH}. Model đang chạy random weights!")
face_model.eval()
# ==========================================
# 2. DATABASE & DATA TRANSFORMS
# ==========================================
DB_FILE = "user_db.pt"
db_lock = threading.Lock()

def load_db():
    if os.path.exists(DB_FILE):
        try: return torch.load(DB_FILE)
        except: return {}
    return {}

face_database = load_db()

def save_db_safe():
    with db_lock:
        torch.save(face_database, DB_FILE)

# Transform chuẩn cho Face Model (Sau khi đã crop)
tf_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
tf_depth = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ==========================================
# 3. CÁC HÀM XỬ LÝ ẢNH (CORE LOGIC)
# ==========================================

def decode_image(request_data):
    """Giải mã base64 thành ảnh PIL RGB"""
    if "," in request_data['image']:
        image_data = request_data['image'].split(",")[1]
    else:
        image_data = request_data['image']
    img_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def generate_depth_full(rgb_img_pil):
    """
    Tạo Depth map cho TOÀN BỘ bức ảnh (Full Context)
    """
    w_orig, h_orig = rgb_img_pil.size
    
    # 1. Resize về kích thước chuẩn của MiDaS (384) để inference tốt nhất
    img_resized = rgb_img_pil.resize((384, 384))
    img_cv = np.array(img_resized)
    
    # 2. Inference
    input_batch = midas_transforms(img_cv).to(DEVICE)
    with torch.no_grad():
        prediction = midas(input_batch)
        
        # 3. Resize kết quả về đúng kích thước ảnh gốc ban đầu
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h_orig, w_orig), 
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
    depth_numpy = prediction.cpu().numpy()
    
    # 4. Normalize về 0-255 (Grayscale)
    depth_min, depth_max = depth_numpy.min(), depth_numpy.max()
    if depth_max - depth_min > 1e-5:
        depth_normalized = (depth_numpy - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth_numpy)
        
    depth_pil = Image.fromarray((depth_normalized * 255).astype("uint8"), mode="L")
    return depth_pil

def crop_face_sync(rgb_full, depth_full):
    """
    Dùng MTCNN tìm mặt trên ảnh RGB, sau đó cắt cả RGB và Depth theo cùng tọa độ
    """
    try:
        # Detect khuôn mặt
        boxes, _ = mtcnn.detect(rgb_full)
        
        if boxes is None or len(boxes) == 0:
            return None, None

        # Lấy khuôn mặt đầu tiên (to nhất)
        box = boxes[0]
        left, top, right, bottom = map(int, box)
        
        # Xử lý biên (padding) để không bị lỗi nếu tọa độ âm
        w, h = rgb_full.size
        left, top = max(0, left), max(0, top)
        right, bottom = min(w, right), min(h, bottom)

        # Cắt đồng bộ
        rgb_face = rgb_full.crop((left, top, right, bottom))
        depth_face = depth_full.crop((left, top, right, bottom))
        
        # Resize về 224x224 chuẩn cho Face Model
        rgb_face = rgb_face.resize((224, 224), Image.BILINEAR)
        depth_face = depth_face.resize((224, 224), Image.BILINEAR)
        
        return rgb_face, depth_face

    except Exception as e:
        print(f"❌ Lỗi cắt mặt: {e}")
        return None, None

def get_embedding(rgb_face, depth_face):
    """Lấy vector đặc trưng từ ảnh mặt đã cắt"""
    rgb_t = tf_rgb(rgb_face).unsqueeze(0).to(DEVICE)
    depth_t = tf_depth(depth_face).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        emb = face_model(rgb_t, depth_t)
    return emb

# ==========================================
# 4. FLASK ROUTES
# ==========================================
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name', 'Unknown')
    
    # 1. Giải mã ảnh gốc từ Client
    rgb_full = decode_image(data)
    
    # 2. Tạo Depth Full (QUAN TRỌNG: Làm trước khi cắt)
    depth_full = generate_depth_full(rgb_full)
    
    # 3. Tìm và Cắt mặt đồng bộ
    rgb_face, depth_face = crop_face_sync(rgb_full, depth_full)
    
    if rgb_face is None:
         return jsonify({"message": "⚠️ Không tìm thấy khuôn mặt! Hãy đứng chính diện.", "status": "error"})

    # 4. Lấy Embedding
    emb = get_embedding(rgb_face, depth_face)
    
    # 5. Lưu vào DB an toàn
    if name not in face_database:
        face_database[name] = []
    face_database[name].append(emb.cpu()) # Lưu ở CPU để tiết kiệm VRAM
    save_db_safe()

    return jsonify({"message": f"✅ Đã đăng ký: {name}", "status": "success"})

@app.route('/identify', methods=['POST'])
def identify():
    data = request.json
    
    # 1. Giải mã & Xử lý pipeline
    rgb_full = decode_image(data)
    depth_full = generate_depth_full(rgb_full)
    rgb_face, depth_face = crop_face_sync(rgb_full, depth_full)
    
    if rgb_face is None:
        # Trả về ảnh gốc nếu không thấy mặt
        return jsonify({"message": "⚠️ Không tìm thấy mặt!", "rgb_url": data['image'], "depth_url": ""})

    # 2. Lấy vector người lại
    unknown_emb = get_embedding(rgb_face, depth_face)
    unknown_emb_cpu = unknown_emb.cpu()

    # 3. So sánh Vectorized (Nhanh)
    best_name = "Người lạ"
    max_similarity = -1.0
    THRESHOLD = 0.6  # Ngưỡng nhận diện (Cosine Similarity)

    # Gom data để nhân ma trận
    all_names = []
    all_vecs = []
    for name, vec_list in face_database.items():
        for vec in vec_list:
            all_names.append(name)
            all_vecs.append(vec)
    
    if len(all_vecs) > 0:
        # Tạo ma trận DB: (N_samples, 512)
        db_matrix = torch.cat(all_vecs, dim=0)
        
        # Tính Cosine: (1, 512) x (512, N) -> (1, N)
        sim_scores = torch.mm(unknown_emb_cpu, db_matrix.T)
        
        # Lấy điểm cao nhất
        max_val, max_idx = torch.max(sim_scores, dim=1)
        max_similarity = max_val.item()
        
        print(f"🔍 Max Score: {max_similarity:.4f} - Ứng viên: {all_names[max_idx.item()]}")
        
        if max_similarity > THRESHOLD:
            best_name = all_names[max_idx.item()]

    # 4. Chuẩn bị ảnh Depth hiển thị (Convert sang base64)
    buffered = io.BytesIO()
    depth_face.save(buffered, format="JPEG") # Trả về ảnh Depth đã cắt mặt
    depth_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    msg = f"✅ Xin chào: {best_name}" if max_similarity > THRESHOLD else "❌ Không nhận ra"
    confidence = round(max_similarity * 100, 2)

    return jsonify({
        "message": f"{msg} (Độ giống: {confidence}%)",
        "depth_url": f"data:image/jpeg;base64,{depth_b64}",
        "rgb_url": data['image']
    })

if __name__ == '__main__':
    # Chạy App
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)