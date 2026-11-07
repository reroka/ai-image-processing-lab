import io
import os
import tempfile
import urllib.request
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# 嘗試載入 MediaPipe，若環境未安裝則自動啟用 OpenCV Haar-Cascade 備援
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection")
except ImportError:
    HAS_MEDIAPIPE = False

# ==================== 頁面基本設定 ====================
st.set_page_config(
    page_title="AI 輔助互動式影像處理與人臉分析工具",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ 互動式影像處理與人臉分析系統")
st.caption("支援多色彩空間轉換、同步直方圖分析、自訂卷積核空間濾波 (含 PSNR 評估) 與深度學習人臉偵測")

# ==================== 核心演算法函式 ====================
def rgb_to_yuv_channels(img_rgb: np.ndarray):
    """
    依 ITU-R BT.601 標準將 RGB 影像轉換為 Y, U, V 三個獨立分量 (0-255 uint8)
    """
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0
    v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0

    y = np.clip(y, 0, 255).astype(np.uint8)
    u = np.clip(u, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    return y, u, v

def calculate_mse_psnr(original: np.ndarray, processed: np.ndarray):
    """
    計算原始影像與濾波後影像之均方誤差 (MSE) 與峰值信噪比 (PSNR)
    """
    orig_f = original.astype(np.float64)
    proc_f = processed.astype(np.float64)
    mse = np.mean((orig_f - proc_f) ** 2)
    if mse == 0:
        return 0.0, float("inf")
    max_pixel = 255.0
    psnr = 10.0 * np.log10((max_pixel ** 2) / mse)
    return mse, psnr

def plot_histogram(channel_data: np.ndarray, title: str, color: str = "steelblue"):
    """
    繪製單一通道或 RGB 三通道之像素強度分布直方圖 (橫軸 0-255，縱軸像素數量)
    """
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=100)
    if channel_data.ndim == 2:
        hist = cv2.calcHist([channel_data], [0], None, [256], [0, 256]).flatten()
        ax.fill_between(range(256), hist, color=color, alpha=0.4)
        ax.plot(range(256), hist, color=color, linewidth=1.5, label=title)
    else:
        colors = [("#e74c3c", "Red (R)"), ("#2ecc71", "Green (G)"), ("#3498db", "Blue (B)")]
        for idx, (c_hex, label) in enumerate(colors):
            hist = cv2.calcHist([channel_data], [idx], None, [256], [0, 256]).flatten()
            ax.plot(range(256), hist, color=c_hex, linewidth=1.3, label=label)
        # 疊加灰階強度曲線
        gray = cv2.cvtColor(channel_data, cv2.COLOR_RGB2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        ax.plot(range(256), hist_gray, color="black", linestyle="--", linewidth=1.2, label="Intensity")

    ax.set_title(f"{title} - Histogram", fontsize=11, fontweight="bold")
    ax.set_xlabel("Pixel Intensity (0 - 255)", fontsize=10)
    ax.set_ylabel("Pixel Count", fontsize=10)
    ax.set_xlim([0, 255])
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig

def get_preset_kernel(name: str, size: int) -> np.ndarray:
    """
    提供預設之 3x3 或 5x5 卷積核矩陣
    """
    if size == 3:
        presets = {
            "自訂矩陣 (Custom)": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
            "單位矩陣 (Identity)": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
            "邊緣銳化 (Sharpen)": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
            "高斯平滑 (Gaussian Blur)": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float),
            "平均值模糊 (Box Blur)": np.ones((3, 3), dtype=float),
            "拉普拉斯邊緣 (Laplacian)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float),
            "浮雕效果 (Emboss)": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float),
        }
    else:
        presets = {
            "自訂矩陣 (Custom)": np.eye(5, dtype=float),
            "單位矩陣 (Identity)": np.pad([[1.0]], 2, mode="constant"),
            "邊緣銳化 (Sharpen)": np.array([
                [0,  0, -1,  0,  0],
                [0, -1, -2, -1,  0],
                [-1, -2, 17, -2, -1],
                [0, -1, -2, -1,  0],
                [0,  0, -1,  0,  0]
            ], dtype=float),
            "高斯平滑 (Gaussian Blur)": np.array([
                [1,  4,  6,  4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1,  4,  6,  4, 1]
            ], dtype=float),
            "平均值模糊 (Box Blur)": np.ones((5, 5), dtype=float),
            "拉普拉斯邊緣 (Laplacian)": np.array([
                [0,  0, -1,  0,  0],
                [0, -1, -2, -1,  0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1,  0],
                [0,  0, -1,  0,  0]
            ], dtype=float),
            "浮雕效果 (Emboss)": np.array([
                [-2, -1, -1,  0,  0],
                [-1, -1,  0,  1,  0],
                [-1,  0,  1,  1,  1],
                [ 0, -1,  1,  1,  1],
                [ 0,  0,  1,  1,  2]
            ], dtype=float),
        }
    return presets.get(name, np.eye(size, dtype=float))

@st.cache_resource
def load_haar_cascade():
    """
    解決 Windows 中文路徑導致 OpenCV C++ CascadeClassifier 讀取失敗，
    以及 OpenCV 5.0 可能缺少內建 XML 檔案的問題。
    """
    xml_name = "haarcascade_frontalface_default.xml"
    # 存放在純英文路徑的系統暫存目錄 (C:\Users\User\AppData\Local\Temp)
    temp_xml_path = os.path.join(tempfile.gettempdir(), xml_name)

    if not os.path.exists(temp_xml_path) or os.path.getsize(temp_xml_path) == 0:
        local_dir = getattr(cv2.data, "haarcascades", "") if hasattr(cv2, "data") else ""
        local_path = os.path.join(local_dir, xml_name)

        if local_path and os.path.exists(local_path):
            # 使用支援中文路徑的 Python open() 讀取並寫入純英文暫存路徑
            with open(local_path, "rb") as src, open(temp_xml_path, "wb") as dst:
                dst.write(src.read())
        else:
            # 若 OpenCV 5.0 環境未附帶 XML，自動從 OpenCV 官方 GitHub 下載
            url = f"https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/{xml_name}"
            urllib.request.urlretrieve(url, temp_xml_path)

    return cv2.CascadeClassifier(temp_xml_path)

# ==================== 側邊欄：功能 1 影像載入 ====================
st.sidebar.header("📂 1. 影像檔案載入")
uploaded_file = st.sidebar.file_uploader(
    "請上傳影像檔案 (PNG, JPG, JPEG, BMP)",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_size_kb = len(file_bytes) / 1024.0
    pil_img = Image.open(io.BytesIO(file_bytes))
    orig_mode = pil_img.mode
    orig_format = pil_img.format if pil_img.format else uploaded_file.name.split(".")[-1].upper()
    raw_array = np.array(pil_img)
    img_rgb = np.array(pil_img.convert("RGB"))
    file_name = uploaded_file.name
else:
    st.sidebar.info("💡 目前使用內建測試影像，請於上方上傳自訂圖片。")
    # 產生內建高對比彩色幾何測試圖
    h, w = 360, 480
    img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img_rgb, (30, 30), (200, 180), (235, 64, 52), -1)
    cv2.circle(img_rgb, (340, 120), 75, (46, 204, 113), -1)
    cv2.rectangle(img_rgb, (80, 210), (420, 330), (52, 152, 219), -1)
    cv2.putText(img_rgb, "AI Image Lab", (110, 285), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    raw_array = img_rgb.copy()
    file_size_kb = img_rgb.nbytes / 1024.0
    orig_mode = "RGB"
    orig_format = "GENERATED (BMP/RGB)"
    file_name = "default_sample.png"

height, width = img_rgb.shape[:2]
channels = 1 if raw_array.ndim == 2 else raw_array.shape[2]

# ==================== 分頁導覽 ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 1. 影像載入與基礎解析",
    "🎨 2. 多色彩空間與直方圖",
    "🧮 3. 自訂卷積濾波與 PSNR",
    "👤 4. 人臉偵測模組"
])

# -------------------- Tab 1: 影像載入與基礎解析 --------------------
with tab1:
    col_img, col_meta = st.columns([1.3, 1])
    with col_img:
        st.subheader("原始輸入影像預覽")
        st.image(img_rgb, caption=f"{file_name} ({width}×{height})", use_container_width=True)

    with col_meta:
        st.subheader("影像基本中繼資訊 (Metadata)")
        m1, m2 = st.columns(2)
        m1.metric("影像解析度 (Width × Height)", f"{width} × {height} px")
        m2.metric("色彩維度 / 通道數", f"{channels} Channels ({orig_mode})")
        m3, m4 = st.columns(2)
        m3.metric("檔案大小 (File Size)", f"{file_size_kb:.2f} KB")
        m4.metric("檔案格式 (Format)", orig_format)

        st.markdown("**詳細陣列與統計屬性**")
        meta_df = pd.DataFrame({
            "屬性項目": ["檔案名稱", "張量維度 (Shape)", "資料型別 (Dtype)", "總像素數", "像素強度範圍 (Min ~ Max)", "平均亮度 (Mean)"],
            "數值內容": [
                file_name,
                str(raw_array.shape),
                str(raw_array.dtype),
                f"{width * height:,} px",
                f"{int(raw_array.min())} ~ {int(raw_array.max())}",
                f"{raw_array.mean():.2f}"
            ]
        })
        st.dataframe(meta_df, use_container_width=True, hide_index=True)

# -------------------- Tab 2: 多色彩空間轉換與同步直方圖 --------------------
with tab2:
    st.subheader("多色彩空間轉換與動態直方圖分析")
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        view_mode = st.selectbox(
            "選擇檢視色彩通道模式：",
            [
                "原始影像 (Original RGB)",
                "RGB - R 通道 (Red)",
                "RGB - G 通道 (Green)",
                "RGB - B 通道 (Blue)",
                "YUV - Y 亮度分量 (Luminance)",
                "YUV - U 色度分量 (Chrominance U)",
                "YUV - V 色度分量 (Chrominance V)"
            ]
        )
    with ctrl_col2:
        rgb_tint = st.checkbox("R/G/B 通道以單色偽彩顯示", value=False, help="勾選後 R/G/B 分量圖將保留對應色彩，直方圖仍維持 0-255 單通道強度統計")

    y_ch, u_ch, v_ch = rgb_to_yuv_channels(img_rgb)
    zeros = np.zeros_like(y_ch)

    if view_mode == "原始影像 (Original RGB)":
        display_img = img_rgb
        hist_data = img_rgb
        hist_color = "black"
    elif view_mode == "RGB - R 通道 (Red)":
        r_ch = img_rgb[:, :, 0]
        display_img = np.stack([r_ch, zeros, zeros], axis=-1) if rgb_tint else r_ch
        hist_data = r_ch
        hist_color = "#e74c3c"
    elif view_mode == "RGB - G 通道 (Green)":
        g_ch = img_rgb[:, :, 1]
        display_img = np.stack([zeros, g_ch, zeros], axis=-1) if rgb_tint else g_ch
        hist_data = g_ch
        hist_color = "#2ecc71"
    elif view_mode == "RGB - B 通道 (Blue)":
        b_ch = img_rgb[:, :, 2]
        display_img = np.stack([zeros, zeros, b_ch], axis=-1) if rgb_tint else b_ch
        hist_data = b_ch
        hist_color = "#3498db"
    elif view_mode == "YUV - Y 亮度分量 (Luminance)":
        display_img = y_ch
        hist_data = y_ch
        hist_color = "#34495e"
    elif view_mode == "YUV - U 色度分量 (Chrominance U)":
        display_img = u_ch
        hist_data = u_ch
        hist_color = "#2980b9"
    else:
        display_img = v_ch
        hist_data = v_ch
        hist_color = "#8e44ad"

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.image(display_img, caption=f"目前檢視：{view_mode}", use_container_width=True, clamp=True)
    with v_col2:
        fig_hist = plot_histogram(hist_data, view_mode, color=hist_color)
        st.pyplot(fig_hist)
        plt.close(fig_hist)

# -------------------- Tab 3: 自訂卷積核空間濾波與 PSNR --------------------
with tab3:
    st.subheader("自訂卷積核 (Kernel Matrix) 空間濾波與量化品質評估")
    k_col1, k_col2 = st.columns([1, 1.6])

    with k_col1:
        k_size = st.radio("1. 選擇卷積核維度：", [3, 5], format_func=lambda x: f"{x} × {x} 矩陣", horizontal=True)
        preset_name = st.selectbox(
            "2. 選擇預設濾波器範本 (可於下方表格直接修改數值)：",
            [
                "邊緣銳化 (Sharpen)",
                "高斯平滑 (Gaussian Blur)",
                "平均值模糊 (Box Blur)",
                "拉普拉斯邊緣 (Laplacian)",
                "浮雕效果 (Emboss)",
                "單位矩陣 (Identity)",
                "自訂矩陣 (Custom)"
            ]
        )
        normalize_flag = st.checkbox("啟用正規化 (Normalization，自動除以權重總和)", value=True)
        add_offset_128 = st.checkbox("零和邊緣/浮雕核加上 +128 灰階偏移", value=False)

        init_matrix = get_preset_kernel(preset_name, k_size)
        df_init = pd.DataFrame(
            init_matrix,
            columns=[f"C{j+1}" for j in range(k_size)],
            index=[f"R{i+1}" for i in range(k_size)]
        )

        st.markdown("**3. 互動式卷積核權重編輯器 (點擊儲存格可直接修改)：**")
        edited_df = st.data_editor(df_init, use_container_width=True, num_rows="fixed")

        # 防呆機制：處理 NaN 或非數值輸入
        kernel_matrix = edited_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        kernel_sum = np.sum(kernel_matrix)

        if normalize_flag:
            if np.isclose(kernel_sum, 0.0, atol=1e-7):
                st.warning("⚠️ 防呆提示：目前卷積核權重總和為 0（如拉普拉斯或零和邊緣核），系統已自動略過除以零正規化，以防止數值溢位！")
                effective_kernel = kernel_matrix
            else:
                effective_kernel = kernel_matrix / kernel_sum
                st.caption(f"✅ 已套用正規化：原始權重總和 = `{kernel_sum:.4f}`")
        else:
            effective_kernel = kernel_matrix
            st.caption(f"ℹ️ 未正規化：目前權重總和 = `{kernel_sum:.4f}`")

    with k_col2:
        # 執行空間域二維卷積運算
        filtered_f = cv2.filter2D(img_rgb.astype(np.float64), ddepth=-1, kernel=effective_kernel, borderType=cv2.BORDER_REFLECT)
        if add_offset_128:
            filtered_f = filtered_f + 128.0
        filtered_img = np.clip(np.round(filtered_f), 0, 255).astype(np.uint8)

        mse_val, psnr_val = calculate_mse_psnr(img_rgb, filtered_img)

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("均方誤差 (MSE)", f"{mse_val:.2f}")
        m_col2.metric("峰值信噪比 (PSNR)", "∞ dB (完全相同)" if np.isinf(psnr_val) else f"{psnr_val:.2f} dB")
        m_col3.metric("卷積核大小", f"{k_size} × {k_size}")

        img_c1, img_c2 = st.columns(2)
        with img_c1:
            st.image(img_rgb, caption="原始影像 (Original)", use_container_width=True)
        with img_c2:
            st.image(filtered_img, caption=f"濾波後影像 ({preset_name})", use_container_width=True)

# -------------------- Tab 4: 人臉偵測模組 --------------------
with tab4:
    st.subheader("智慧人臉偵測與信心度標註")
    fd_col1, fd_col2 = st.columns([1, 2.2])

    with fd_col1:
        engine_options = ["MediaPipe Face Detection (輕量深度學習)"] if HAS_MEDIAPIPE else []
        engine_options.append("OpenCV Haar-Cascade (經典特徵辨識)")
        detector_type = st.selectbox("選擇偵測演算法模型：", engine_options)
        conf_threshold = st.slider("最低推論信心度門檻 (Confidence Threshold)：", 0.10, 0.95, 0.50, 0.05)

    annotated_img = img_rgb.copy()
    detections_list = []

    if detector_type.startswith("MediaPipe") and HAS_MEDIAPIPE:
        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(model_selection=1, min_detection_confidence=conf_threshold) as face_detection:
            results = face_detection.process(img_rgb)
            if results.detections:
                for idx, det in enumerate(results.detections, start=1):
                    score = float(det.score[0])
                    bboxC = det.location_data.relative_bounding_box
                    x = max(0, int(bboxC.xmin * width))
                    y = max(0, int(bboxC.ymin * height))
                    w_box = min(width - x, int(bboxC.width * width))
                    h_box = min(height - y, int(bboxC.height * height))

                    detections_list.append({
                        "人臉編號": f"Face #{idx}",
                        "信心度 (Confidence)": f"{score:.4f}",
                        "左上座標 (X, Y)": f"({x}, {y})",
                        "邊界框尺寸 (W × H)": f"{w_box} × {h_box}"
                    })
                    cv2.rectangle(annotated_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_img,
                        f"#{idx}: {score:.2f}",
                        (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
    else:
        gray_for_face = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        face_cascade = load_haar_cascade()

        if face_cascade.empty():
            st.error("❌ Haar-Cascade 模型載入失敗，請確認網路連線或暫存目錄權限。")
            rects, level_weights = [], []
        else:
            rects, reject_levels, level_weights = face_cascade.detectMultiScale3(
                gray_for_face,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30),
                outputRejectLevels=True
            )

        if len(rects) > 0:
            for idx, ((x, y, w_box, h_box), weight) in enumerate(zip(rects, level_weights), start=1):
                # 將 Haar 階段權重映射為 [0, 1] 推論信心度
                score = float(1.0 / (1.0 + np.exp(-0.5 * float(weight))))
                if score < conf_threshold:
                    continue
                detections_list.append({
                    "人臉編號": f"Face #{idx}",
                    "信心度 (Confidence)": f"{score:.4f}",
                    "左上座標 (X, Y)": f"({x}, {y})",
                    "邊界框尺寸 (W × H)": f"{w_box} × {h_box}"
                })
                cv2.rectangle(annotated_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(
                    annotated_img,
                    f"#{idx}: {score:.2f}",
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    with fd_col1:
        st.metric("偵測到的總人臉數", f"{len(detections_list)} 人")
        if detections_list:
            st.dataframe(pd.DataFrame(detections_list), use_container_width=True, hide_index=True)
        else:
            st.info("未偵測到符合信心門檻的人臉，可嘗試調降門檻或上傳含有人臉之照片。")

    with fd_col2:
        st.image(annotated_img, caption=f"人臉偵測結果 (共偵測到 {len(detections_list)} 張人臉)", use_container_width=True)