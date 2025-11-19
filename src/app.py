import streamlit as st
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import io

# Đảm bảo rằng các file .py này ở cùng thư mục với app.py
try:
    from models import MetaLearnerMLP
    from feature_extractor_finetuned import FinetunedXceptionFeatureExtractor, FinetunedEfficientNetB3FeatureExtractor
except ImportError:
    st.error("Lỗi: Không tìm thấy file `models.py` hoặc `feature_extractor_finetuned.py`. "
             "Hãy đảm bảo chúng ở cùng thư mục với `app.py`.")
    st.stop()

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Trình Phát Hiện Deepfake",
    layout="wide"
)

# --- Load Model (Cache lại để chỉ load 1 lần) ---
# Sử dụng @st.cache_resource để load model một lần duy nhất
@st.cache_resource
def load_models(ckpt_path='ensemble_finetuned_best.pth'):
    """
    Load checkpoint và khởi tạo tất cả các model.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Tải checkpoint. 
        # Cảnh báo: weights_only=False có thể không an toàn nếu file ckpt
        # đến từ nguồn không đáng tin cậy vì nó unpickle object 'scaler'.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file checkpoint '{ckpt_path}'. "
                 "Hãy đảm bảo file này ở đúng vị trí.")
        return None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"Lỗi khi load checkpoint: {e}")
        st.warning("Lưu ý: Checkpoint này chứa 'scaler' (một đối tượng Python), "
                   "vì vậy cần `weights_only=False`. "
                   "Hãy đảm bảo bạn tin tưởng nguồn gốc của file checkpoint này.")
        return None, None, None, None, None, None, None

    # 1. Khởi tạo Feature Extractors
    try:
        xcp = FinetunedXceptionFeatureExtractor(ckpt['xception_ckpt']).to(device).eval()
        eff = FinetunedEfficientNetB3FeatureExtractor(ckpt['efficientnet_ckpt']).to(device).eval()
    except KeyError as e:
        st.error(f"Lỗi: Không tìm thấy key '{e}' trong checkpoint. "
                 "Checkpoint có thể bị hỏng hoặc không đúng định dạng.")
        return None, None, None, None, None, None, None
    except FileNotFoundError as e:
        st.error(f"Lỗi: Không tìm thấy file model con: {e.filename}. "
                 "Đường dẫn trong 'xception_ckpt' hoặc 'efficientnet_ckpt' có thể bị sai.")
        return None, None, None, None, None, None, None

    # 2. Khởi tạo Meta Learner (MLP)
    mlp = MetaLearnerMLP(input_dim=ckpt['input_dim']).to(device)
    mlp.load_state_dict(ckpt['model_state_dict'])
    mlp.eval()

    # 3. Lấy Scaler và Indices
    scaler = ckpt['scaler']
    selected_indices = ckpt['selected_indices']

    # 4. Định nghĩa Image Transform
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return xcp, eff, mlp, scaler, selected_indices, tfm, device

# --- Hàm dự đoán ---
def predict_image(image_bytes, xcp, eff, mlp, scaler, selected_indices, tfm, device):
    """
    Chạy toàn bộ pipeline dự đoán cho một ảnh đầu vào.
    """
    try:
        # Mở ảnh từ bytes và chuyển sang RGB
        img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        st.error(f"Lỗi khi đọc ảnh: {e}")
        return None, None, None

    # 1. Áp dụng transform
    img_tensor = tfm(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        # 2. Trích xuất features
        f_xcp = xcp(img_tensor).cpu().numpy()
        f_eff = eff(img_tensor).cpu().numpy()
        
        # 3. Combine features
        features = np.hstack([f_xcp, f_eff])

        # 4. Normalize & Select features
        X = scaler.transform(features)
        X = X[:, selected_indices]
        
        # 5. Predict
        out = mlp(torch.FloatTensor(X).to(device))
        probs = torch.softmax(out, dim=1)
        
        # Lấy xác suất của class 1 (FAKE)
        prob_fake = probs[:, 1].item() 
        
    # Xác định nhãn
    label = "FAKE" if prob_fake > 0.5 else "REAL"
    
    return img_pil, label, prob_fake

# --- Giao diện Streamlit ---
st.title("Trình Phát Hiện Ảnh Deepfake")
st.write("Upload một ảnh để kiểm tra xem đó là ảnh **REAL** (thật) hay **FAKE** (sản phẩm của deepfake).")

# Sidebar
st.sidebar.title("Thông tin Model")
st.sidebar.info(
    """
    Ứng dụng này sử dụng mô hình ensemble (kết hợp) để dự đoán:
    
    1.  **Feature Extractors:**
        * Finetuned Xception
        * Finetuned EfficientNet-B3
    2.  **Meta-Learner:**
        * Một mô hình MLP (Multi-Layer Perceptron) học cách kết hợp các đặc trưng từ hai model trên.
    
    Mô hình được huấn luyện để phân biệt ảnh thật và ảnh giả (deepfake).
    """
)
# Load models (chỉ chạy lần đầu)
models = load_models()

if all(m is not None for m in models):
    xcp, eff, mlp, scaler, selected_indices, tfm, device = models
    
    # Khu vực upload ảnh
    uploaded_file = st.file_uploader(
        "Chọn một file ảnh (jpg, jpeg, png)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Lấy image bytes
        image_bytes = uploaded_file.getvalue()
        
        # Hiển thị spinner trong khi xử lý
        with st.spinner('Đang phân tích ảnh...'):
            img_pil, label, prob_fake = predict_image(
                image_bytes, xcp, eff, mlp, scaler, selected_indices, tfm, device
            )

        if img_pil:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img_pil, caption="Ảnh đã upload", use_column_width=True)
            
            with col2:
                st.subheader("Kết Quả Phân Tích")
                
                if label == "FAKE":
                    st.error(f"Dự đoán: FAKE")
                    st.write(f"Mô hình khá chắc chắn đây là ảnh **FAKE**.")
                else:
                    st.success(f"Dự đoán: REAL")
                    st.write(f"Mô hình cho rằng đây là ảnh **REAL**.")
                
                st.write("---")
                
                # Hiển thị confidence
                st.write("Confidence Score (cho FAKE):")
                st.progress(prob_fake)
                st.metric(
                    label="Xác suất là FAKE", 
                    value=f"{prob_fake * 100:.2f} %"
                )
                
                st.write(f"Giá trị *raw probability* (cho class FAKE) là **{prob_fake:.4f}**. "
                         "Nếu > 0.5, ảnh được phân loại là FAKE.")

else:
    st.error("Không thể khởi tạo mô hình. Vui lòng kiểm tra console để biết chi tiết.")